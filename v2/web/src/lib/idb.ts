/**
 * Minimal promise wrapper over IndexedDB.
 *
 * History data used to live in localStorage, which is a synchronous ~5 MB
 * store. That cannot hold full-size source images plus optimization results:
 * a single optimized entry is roughly 1 MB, and a phone photo on its own can
 * exceed the entire quota. The symptom was QuotaExceededError, plus entries
 * silently losing their source image and becoming impossible to re-optimize
 * after a reload.
 *
 * IndexedDB is asynchronous and has a far larger per-origin quota, so the
 * heavy data lives here instead.
 */

const DB_NAME = "typewriter-art";
const STORE_NAME = "kv";

/**
 * Version 2 wipes the store on upgrade. This app is not deployed anywhere and
 * no legacy shapes are supported any more (localStorage-era `resultMeta`
 * entries, thumbnail-only sources), so a clean slate replaces migration code.
 */
const DB_VERSION = 2;

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (!dbPromise) {
    dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      if (typeof indexedDB === "undefined") {
        reject(new Error("IndexedDB is unavailable"));
        return;
      }
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        if (request.result.objectStoreNames.contains(STORE_NAME)) {
          request.result.deleteObjectStore(STORE_NAME);
        }
        request.result.createObjectStore(STORE_NAME);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
      request.onblocked = () => reject(new Error("IndexedDB open blocked"));
    });
    // Never cache a rejection — a later attempt may succeed (e.g. after a
    // blocked open is released).
    dbPromise.catch(() => {
      dbPromise = null;
    });
  }
  return dbPromise;
}

function run<T>(
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        let request: IDBRequest<T>;
        try {
          request = fn(db.transaction(STORE_NAME, mode).objectStore(STORE_NAME));
        } catch (err) {
          reject(err instanceof Error ? err : new Error(String(err)));
          return;
        }
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
      })
  );
}

// The DOM typings declare `IDBRequest<T>` as invariant and give `get`, `put`
// and `delete` different result types, so each call site narrows explicitly.
export function idbGet<T>(key: string): Promise<T | undefined> {
  return run<T | undefined>("readonly", (store) =>
    store.get(key) as unknown as IDBRequest<T | undefined>
  );
}

export function idbSet(key: string, value: unknown): Promise<void> {
  return run<void>("readwrite", (store) =>
    store.put(value, key) as unknown as IDBRequest<void>
  );
}

export function idbRemove(key: string): Promise<void> {
  return run<void>("readwrite", (store) =>
    store.delete(key) as unknown as IDBRequest<void>
  );
}
