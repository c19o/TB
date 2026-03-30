#!/usr/bin/env python3
"""Patch v2_cross_generator.py to flush COO→CSR per RIGHT_CHUNK inside gpu_batch_cross"""
import sys

filepath = '/workspace/v3.3/v2_cross_generator.py'
with open(filepath, 'r') as f:
    lines = f.readlines()

output = []
in_batch_cross = False
found_accum = False
found_init = False
found_return = False
found_collect = False
skip_until_return = False

i = 0
while i < len(lines):
    line = lines[i]

    # 1. Add _local_csr_chunks init after all_data = []
    if '    all_data = []' in line and not found_init and i > 300:
        output.append(line)
        output.append('    _local_csr_chunks = []  # CSR chunks flushed per RIGHT_CHUNK\n')
        found_init = True
        i += 1
        continue

    # 2. Replace accumulation block: detect "if c_names:" inside gpu_batch_cross
    if '        if c_names:' in line and not found_accum and i > 380 and i < 430:
        # Find the extent of this block (until "del right_mat_chunk")
        j = i + 1
        while j < len(lines) and 'del right_mat_chunk' not in lines[j]:
            j += 1
        # j now points to "del right_mat_chunk" line
        # Also include gc.collect() line after it
        gc_line = j + 1

        # Write replacement
        output.append('        if c_names:\n')
        output.append('            # FLUSH per RIGHT_CHUNK: convert COO to CSR immediately\n')
        output.append('            all_names.extend(c_names)\n')
        output.append('            if c_rows:\n')
        output.append('                _flush_r = np.concatenate(c_rows)\n')
        output.append('                _flush_c = np.concatenate(c_cols)\n')
        output.append('                _flush_d = np.concatenate(c_data)\n')
        output.append('                _flush_c_local = _flush_c - current_offset\n')
        output.append('                _flush_ncols = int(_flush_c_local.max()) + 1 if len(_flush_c_local) > 0 else c_ncols\n')
        output.append('                _flush_csr = sparse.coo_matrix((_flush_d, (_flush_r, _flush_c_local)), shape=(N, _flush_ncols)).tocsr()\n')
        output.append('                _local_csr_chunks.append(_flush_csr)\n')
        output.append('                del _flush_r, _flush_c, _flush_d, _flush_c_local, _flush_csr\n')
        output.append('            current_offset += c_ncols\n')
        output.append('            total_feats += len(c_names)\n')
        output.append('            del c_rows, c_cols, c_data, c_names\n')
        output.append('\n')
        output.append('        del right_mat_chunk, r_arrays_chunk\n')
        output.append('        gc.collect()\n')

        found_accum = True
        # Skip to after gc.collect() line
        i = gc_line + 1
        continue

    # 3. Replace return statement in gpu_batch_cross
    if '    del left_mat' in line and 'n_total_cols = current_offset - col_offset' in lines[i+1] if i+1 < len(lines) else False:
        output.append('    del left_mat\n')
        output.append('    n_total_cols = current_offset - col_offset\n')
        output.append('    # Return CSR chunks if we flushed (memory-safe path)\n')
        output.append('    if _local_csr_chunks:\n')
        output.append('        return all_names, _local_csr_chunks, None, None, n_total_cols\n')
        output.append('    return all_names, all_rows, all_cols, all_data, n_total_cols\n')
        found_return = True
        i += 3  # skip old del, n_total, return lines
        continue

    # 4. Replace _collect_cross to handle CSR chunks
    if 'def _collect_cross(label, names, rows_list, cols_list, data_list, n_new_cols):' in line and not found_collect:
        output.append(line)  # keep the def line
        # Skip the old body until the next "return count" at same indent
        j = i + 1
        indent_level = len(line) - len(line.lstrip())
        while j < len(lines):
            if lines[j].strip() == 'return count' and (len(lines[j]) - len(lines[j].lstrip())) > indent_level:
                break
            j += 1
        # j points to "return count"

        # Write new body
        output.append('        """Convert cross type results to CSR. Handles both CSR-flushed and legacy COO paths."""\n')
        output.append('        nonlocal col_offset, _total_collected\n')
        output.append('        count = len(names)\n')
        output.append('        if count > 0:\n')
        output.append('            all_cross_names.extend(names)\n')
        output.append('            # Check if rows_list contains pre-flushed CSR chunks\n')
        output.append('            if rows_list is not None and isinstance(rows_list, list) and len(rows_list) > 0 and hasattr(rows_list[0], "indptr"):\n')
        output.append('                _csr_chunks.extend(rows_list)\n')
        output.append('                col_offset += n_new_cols\n')
        output.append('                _total_collected += count\n')
        output.append('                gc.collect()\n')
        output.append('            elif rows_list is not None and cols_list is not None and isinstance(rows_list, list) and len(rows_list) > 0:\n')
        output.append('                _r = np.concatenate(rows_list)\n')
        output.append('                _c = np.concatenate(cols_list)\n')
        output.append('                _d = np.concatenate(data_list)\n')
        output.append('                _c_local = _c - col_offset\n')
        output.append('                chunk = sparse.coo_matrix((_d, (_r, _c_local)), shape=(N, n_new_cols)).tocsr()\n')
        output.append('                _csr_chunks.append(chunk)\n')
        output.append('                del _r, _c, _c_local, _d, chunk\n')
        output.append('                gc.collect()\n')
        output.append('                col_offset += n_new_cols\n')
        output.append('                _total_collected += count\n')
        output.append('            else:\n')
        output.append('                col_offset += n_new_cols\n')
        output.append('                _total_collected += count\n')
        pref = ' ' * (indent_level + 8)
        output.append(f'{pref}log(f"    {{label}} crosses: {{count:,}} (total: {{_total_collected:,}})")\n')
        output.append('        return count\n')

        found_collect = True
        i = j + 1  # skip past old "return count"
        continue

    output.append(line)
    i += 1

with open(filepath, 'w') as f:
    f.writelines(output)

print(f'Patches applied: init={found_init}, accum={found_accum}, return={found_return}, collect={found_collect}')
if not all([found_init, found_accum, found_return, found_collect]):
    print('WARNING: Not all patches applied!')
