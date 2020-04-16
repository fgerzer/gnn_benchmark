from pymongo import MongoClient
import os
import time
from collections import namedtuple
from gnn_benchmark.common.definitions import RunState, RunEntry, RunResults

DBPaths = namedtuple("DBPaths", "host database collection")


class RunsDB:
    def __init__(self, db_path: DBPaths):
        if not isinstance(db_path.collection, list):
            collection = [db_path.collection]
        else:
            collection = db_path.collection
        self.collections = []
        for c in collection:
            self.collections.append(MongoClient(db_path.host)[db_path.database][c])
        self._locked = False

    def find(self, filter):
        finds = [c.find(filter) for c in self.collections]

        def find_iter():
            for f in finds:
                for r in f:
                    yield RunEntry.from_dict(r)
        return find_iter()

    def n_runs(self, run_state=None):
        filter = {}
        if run_state is not None:
            filter = {"run_state": run_state}

        return sum([c.count_documents(filter=filter) for c in self.collections])

    def find_finished(self):
        return self.find(filter={"run_state": RunState.finished})

    def ensure_one_collection(self):
        if len(self.collections) > 1:
            raise ValueError(f"Only supporting insertion of new instances when a single DB collection has been "
                             f"defined, but {len(self.collections)} have been defined (with {self.collections}).")
        return self.collections[0]

    def insert_run(self, run_definition, run_state=RunState.pending):
        collection = self.ensure_one_collection()
        run_entry = RunEntry(
            run_state=run_state,
            run_definition=run_definition,
            results=RunResults(),
            id=None
        )
        run_id = collection.insert_one(run_entry.to_dict()).inserted_id
        run_entry.id = run_id
        return run_entry

    def insert_runs(self, run_definitions, run_state=RunState.pending):
        run_entries = []
        for r in run_definitions:
            run_entries.append(self.insert_run(r, run_state))
        return run_entries

    def claim_run(self):
        collection = self.ensure_one_collection()
        job_id = os.environ.get("SLURM_JOB_ID", None)
        update = {'$set': {"run_state": RunState.running}}
        if job_id:
            update["$set"]["job_id"] = job_id
        d = None
        if job_id:
            # We have a job id, i.e. are running on a slurm cluster. If there exists a running job with our job ID,
            # we claim it. This takes care of requeued jobs.
            d = collection.find_one_and_update(
                {"job_id": job_id, "run_state": RunState.running},
                update=update
            )
        if d is None:
            # We now look for any pending job, and claim it.
            d = collection.find_one_and_update({"run_state": RunState.pending}, update=update)
        if d is None:
            return None
        return RunEntry.from_dict(d)

    def submit_result(self, run_entry, results, run_state=RunState.finished):
        collection = self.ensure_one_collection()
        result = collection.find_one_and_update(
            {"_id": run_entry.id},
            update={'$set': {"run_state": run_state, "results": results.to_dict()}}
        )

    def lock(self):
        collection = self.ensure_one_collection()
        # Extremely stupid lock system to avoid multiple runs creating hyperparameters at the same time
        if collection.count_documents(filter={}) > 0:
            return False
        l = collection.find_one_and_update(filter={"currently_creating_runs": True},
                                                update={"$setOnInsert": {"currently_creating_runs": True}},
                                                upsert=True)
        # if l is None, has been newly created; this got the lock
        self._locked = l is None
        return self._locked

    def unlock(self):
        collection = self.ensure_one_collection()
        r = collection.find_one_and_delete(filter={"currently_creating_runs": True})
        self._locked = False

    def wait_for_unlock(self):
        collection = self.ensure_one_collection()
        while collection.find_one(filter={"currently_creating_runs": True}) is not None:
            time.sleep(0.25)

