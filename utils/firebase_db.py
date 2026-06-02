import os
import json
import uuid
import datetime
from config import Config

# Try to import firebase_admin
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

class FirebaseDB:
    def __init__(self):
        self.firebase_active = False
        self.firestore_client = None
        
        # Check if Firebase credentials exist and firebase_admin is installed
        if FIREBASE_AVAILABLE and os.path.exists(Config.FIREBASE_CREDENTIALS):
            try:
                # Initialize Firebase if not already initialized
                if not firebase_admin._apps:
                    cred = credentials.Certificate(Config.FIREBASE_CREDENTIALS)
                    firebase_admin.initialize_app(cred)
                self.firestore_client = firestore.client()
                self.firebase_active = True
                print("[FirebaseDB] Successfully connected to Firebase Firestore.")
            except Exception as e:
                print(f"[FirebaseDB] Error initializing Firebase: {e}. Falling back to local JSON database.")
                self.firebase_active = False
        else:
            if not FIREBASE_AVAILABLE:
                print("[FirebaseDB] firebase-admin package not found. Using local JSON database.")
            else:
                print("[FirebaseDB] Firebase credentials file not found. Using local JSON database.")
            self.firebase_active = False

        if not self.firebase_active:
            self._init_local_db()

    def _init_local_db(self):
        """Initializes the local JSON file database if it doesn't exist."""
        if not os.path.exists(Config.LOCAL_DB_PATH):
            os.makedirs(os.path.dirname(Config.LOCAL_DB_PATH), exist_ok=True)
            with open(Config.LOCAL_DB_PATH, 'w') as f:
                json.dump({}, f)
            print(f"[FirebaseDB] Created local fallback database at {Config.LOCAL_DB_PATH}")

    def _read_local_db(self):
        """Reads data from the local JSON database."""
        self._init_local_db()
        try:
            with open(Config.LOCAL_DB_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_local_db(self, data):
        """Writes data to the local JSON database."""
        try:
            with open(Config.LOCAL_DB_PATH, 'w') as f:
                json.dump(data, f, indent=4, default=str)
        except Exception as e:
            print(f"[FirebaseDB] Error writing to local DB: {e}")

    def add_document(self, collection_name, data):
        """Adds a document to a collection. Generates a random ID."""
        # Convert datetime objects to string representation for compatibility
        data_copy = self._prepare_data(data)
        
        if self.firebase_active:
            try:
                # Add doc to Firestore
                _, doc_ref = self.firestore_client.collection(collection_name).add(data_copy)
                return doc_ref.id
            except Exception as e:
                print(f"[FirebaseDB] Firestore error: {e}. Attempting local write.")
                # Fallback to local write if Firestore fails
                return self._add_local(collection_name, data_copy)
        else:
            return self._add_local(collection_name, data_copy)

    def _add_local(self, collection_name, data):
        db_data = self._read_local_db()
        if collection_name not in db_data:
            db_data[collection_name] = {}
        
        doc_id = str(uuid.uuid4())
        db_data[collection_name][doc_id] = data
        self._write_local_db(db_data)
        return doc_id

    def set_document(self, collection_name, doc_id, data):
        """Sets/overwrites a specific document in a collection."""
        data_copy = self._prepare_data(data)
        
        if self.firebase_active:
            try:
                self.firestore_client.collection(collection_name).document(doc_id).set(data_copy)
                return doc_id
            except Exception as e:
                print(f"[FirebaseDB] Firestore error: {e}. Writing locally.")
                return self._set_local(collection_name, doc_id, data_copy)
        else:
            return self._set_local(collection_name, doc_id, data_copy)

    def _set_local(self, collection_name, doc_id, data):
        db_data = self._read_local_db()
        if collection_name not in db_data:
            db_data[collection_name] = {}
        db_data[collection_name][doc_id] = data
        self._write_local_db(db_data)
        return doc_id

    def get_document(self, collection_name, doc_id):
        """Retrieves a document. Returns a dict or None."""
        if self.firebase_active:
            try:
                doc = self.firestore_client.collection(collection_name).document(doc_id).get()
                if doc.exists:
                    doc_data = doc.to_dict()
                    doc_data['id'] = doc.id
                    return doc_data
                return None
            except Exception as e:
                print(f"[FirebaseDB] Firestore error: {e}. Reading locally.")
                return self._get_local(collection_name, doc_id)
        else:
            return self._get_local(collection_name, doc_id)

    def _get_local(self, collection_name, doc_id):
        db_data = self._read_local_db()
        collection = db_data.get(collection_name, {})
        doc_data = collection.get(doc_id)
        if doc_data:
            doc_data = dict(doc_data)
            doc_data['id'] = doc_id
            return doc_data
        return None

    def get_documents(self, collection_name):
        """Gets all documents from a collection."""
        if self.firebase_active:
            try:
                docs = self.firestore_client.collection(collection_name).stream()
                result = []
                for doc in docs:
                    d = doc.to_dict()
                    d['id'] = doc.id
                    result.append(d)
                return result
            except Exception as e:
                print(f"[FirebaseDB] Firestore error: {e}. Reading locally.")
                return self._get_all_local(collection_name)
        else:
            return self._get_all_local(collection_name)

    def _get_all_local(self, collection_name):
        db_data = self._read_local_db()
        collection = db_data.get(collection_name, {})
        result = []
        for doc_id, doc_data in collection.items():
            d = dict(doc_data)
            d['id'] = doc_id
            result.append(d)
        return result

    def query_documents(self, collection_name, field, value, op='=='):
        """Queries documents in a collection. Default operator is '=='."""
        if self.firebase_active:
            try:
                docs = self.firestore_client.collection(collection_name).where(field, op, value).stream()
                result = []
                for doc in docs:
                    d = doc.to_dict()
                    d['id'] = doc.id
                    result.append(d)
                return result
            except Exception as e:
                print(f"[FirebaseDB] Firestore error: {e}. Falling back to local search.")
                return self._query_local(collection_name, field, value, op)
        else:
            return self._query_local(collection_name, field, value, op)

    def _query_local(self, collection_name, field, value, op):
        db_data = self._read_local_db()
        collection = db_data.get(collection_name, {})
        result = []
        for doc_id, doc_data in collection.items():
            if field in doc_data:
                item_val = doc_data[field]
                match = False
                if op == '==':
                    match = str(item_val) == str(value)
                elif op == '>':
                    match = item_val > value
                elif op == '<':
                    match = item_val < value
                elif op == '>=':
                    match = item_val >= value
                elif op == '<=':
                    match = item_val <= value
                
                if match:
                    d = dict(doc_data)
                    d['id'] = doc_id
                    result.append(d)
        return result

    def delete_document(self, collection_name, doc_id):
        """Deletes a document from a collection."""
        if self.firebase_active:
            try:
                self.firestore_client.collection(collection_name).document(doc_id).delete()
                return True
            except Exception as e:
                print(f"[FirebaseDB] Firestore error: {e}. Deleting locally.")
                return self._delete_local(collection_name, doc_id)
        else:
            return self._delete_local(collection_name, doc_id)

    def _delete_local(self, collection_name, doc_id):
        db_data = self._read_local_db()
        if collection_name in db_data and doc_id in db_data[collection_name]:
            del db_data[collection_name][doc_id]
            self._write_local_db(db_data)
            return True
        return False

    def _prepare_data(self, data):
        """Converts datetime objects in a dict to strings for storage compatibility."""
        res = {}
        for k, v in data.items():
            if isinstance(v, (datetime.datetime, datetime.date)):
                res[k] = v.isoformat()
            else:
                res[k] = v
        return res

# Global database helper instance
db_client = FirebaseDB()
