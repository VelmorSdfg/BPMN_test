import cv2
import numpy as np
import os
import math

def detect_orthogonal_arrows(img_input, output_dir=None):
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input.copy()

    if img is None: return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    output = img.copy()

    # --- 1. Поиск базовых линий ---
    def find_lines_by_direction(mask, direction='h'):
        size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1) if direction == 'h' else (1, size))
        line_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if (w if direction == 'h' else h) < 10: continue
            ends = [(x, y + h // 2), (x + w, y + h // 2)] if direction == 'h' else [(x + w // 2, y),
                                                                                    (x + w // 2, y + h)]
            results.append({'rect': [int(x), int(y), int(w), int(h)], 'ends': ends, 'dir': direction.upper()})
        return results

    raw_segments = find_lines_by_direction(binary, 'h') + find_lines_by_direction(binary, 'v')
    num_segments = len(raw_segments)
    if num_segments == 0: return []

    # --- 2. Построение графа связей ---
    adj = {i: set() for i in range(num_segments)}
    sphere_radius = 15

    for i in range(num_segments):
        seg_a = raw_segments[i]
        for j in range(i + 1, num_segments):
            seg_b = raw_segments[j]
            is_connected = False

            for p_a in seg_a['ends']:
                for p_b in seg_b['ends']:
                    if math.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2) <= sphere_radius:
                        is_connected = True; break
                if is_connected: break

            if not is_connected:
                for p_a in seg_a['ends']:
                    bx, by, bw, bh = seg_b['rect']
                    if (bx - 5 <= p_a[0] <= bx + bw + 5 and by - 5 <= p_a[1] <= by + bh + 5):
                        is_connected = True; break
                if not is_connected:
                    for p_b in seg_b['ends']:
                        ax, ay, aw, ah = seg_a['rect']
                        if (ax - 5 <= p_b[0] <= ax + aw + 5 and ay - 5 <= p_b[1] <= ay + ah + 5):
                            is_connected = True; break
            if is_connected:
                adj[i].add(j); adj[j].add(i)

    # --- 3. Группировка (BFS) ---
    groups = []
    visited = set()
    for i in range(num_segments):
        if i not in visited:
            group = []; queue = [i]; visited.add(i)
            while queue:
                curr = queue.pop(0)
                group.append(raw_segments[curr])
                for n in adj[curr]:
                    if n not in visited: visited.add(n); queue.append(n)
            groups.append(group)

    # --- 4. Анализ и Фильтрация пустых стрелок ---
    arrows_final_data = []

    for g_idx, group in enumerate(groups):
        external_ends = []
        for seg in group:
            for ex, ey in seg['ends']:
                is_internal = False
                neighbor_count = 0
                for other_seg in group:
                    if other_seg is seg: continue
                    ox, oy, ow, oh = other_seg['rect']
                    if (ox - 8 <= ex <= ox + ow + 8 and oy - 8 <= ey <= oy + oh + 8):
                        neighbor_count += 1
                    for p_o in other_seg['ends']:
                        if math.sqrt((ex - p_o[0]) ** 2 + (ey - p_o[1]) ** 2) < 12:
                            is_internal = True; break
                    if is_internal: break

                if neighbor_count > 0: is_internal = True
                if not is_internal: external_ends.append((int(ex), int(ey)))

        # Определение TIP
        best_tip = None
        max_density = -1
        for ex, ey in external_ends:
            r = 15
            roi = binary[max(0, ey - r):ey + r, max(0, ex - r):ex + r]
            density = cv2.countNonZero(roi)
            if density > max_density:
                max_density = density
                best_tip = [ex, ey]

        # STARTS — внешние концы без учета tip
        start_points = [p for p in external_ends if p != best_tip]

        # Записываем только если стрелка не пустая
        if best_tip is not None or len(start_points) > 0:
            arrows_final_data.append({
                "id": f"arrow_{len(arrows_final_data)}", # Пересчитываем ID, чтобы не было дырок
                "tip": best_tip,
                "starts": start_points
            })

            # Отрисовка  валидных стрелок
            if output_dir:
                base_color = (int(np.random.randint(50, 200)), int(np.random.randint(50, 200)), int(np.random.randint(50, 200)))
                for seg in group:
                    x, y, w, h = seg['rect']
                    cv2.rectangle(output, (x, y), (x + w, y + h), base_color, 2)
                if best_tip:
                    cv2.circle(output, (best_tip[0], best_tip[1]), 7, (0, 255, 0), -1)
                for sp in start_points:
                    cv2.circle(output, (sp[0], sp[1]), 5, (0, 0, 255), -1)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, '4_detected_arrows_final.png'), output)

    return arrows_final_data