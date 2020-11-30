def parallel(func, arr:Collection, max_workers:int=None, leave=False):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    max_workers = ifnone(max_workers, defaults.cpus)
    if max_workers<2: results = [func(o,i) for i,o in progress_bar(enumerate(arr), total=len(arr), leave=leave)]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
            results = []
            for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr), leave=leave): 
                results.append(f.result())
    if any([o is not None for o in results]): return results