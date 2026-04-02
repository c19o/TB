import json, sys
data = json.load(sys.stdin)
target = 30534388

found = False
for m in data:
    if m.get('id') == target or m.get('ask_contract_id') == target:
        print(json.dumps(m, indent=2))
        found = True
        break

if not found:
    print(f'Machine {target} not found in current listings')
    print(f'Found {len(data)} Japan 2x4090 machines available')
    if data:
        print('Top 3 by dlperf_per_dphtotal:')
        sorted_data = sorted(data, key=lambda x: x.get('dlperf_per_dphtotal', 0), reverse=True)[:3]
        for m in sorted_data:
            mid = m.get('id')
            perf = m.get('dlperf_per_dphtotal', 0)
            price = m.get('dph_total', 0)
            cores = m.get('cpu_cores', 0)
            ghz = m.get('cpu_ghz', 0)
            cpu_score = cores * ghz
            print(f"  ID {mid}: {perf:.2f} perf/$ @ ${price:.3f}/hr, {cores}c {ghz:.2f}GHz (CPU Score: {cpu_score:.0f})")
