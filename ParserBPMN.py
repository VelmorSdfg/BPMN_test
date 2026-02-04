import os
import asyncio
import xml.etree.ElementTree as ET
from playwright.async_api import async_playwright

# РАСШИРЕННЫЙ СПИСОК КЛАССОВ (NC: 5)
# Разделяем события, так как у них разная толщина границ и геометрия (тонкая, жирная, двойная линия)
CLASSES = ['Task', 'Gateway', 'StartEvent', 'EndEvent', 'IntermediateEvent']

async def process_bpmn_dataset(folder_path):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/bpmn-js/dist/bpmn-viewer.production.min.js"></script>
        <style>
            #canvas { width: 100vw; height: 100vh; background: white; }
            body, html { margin: 0; padding: 0; overflow: hidden; }
            .bjs-powered-by { display: none !important; }
        </style>
    </head>
    <body>
        <div id="canvas"></div>
        <script>
            window.viewer = new BpmnJS({ container: '#canvas' });
            window.getRenderData = async (xml) => {
                try {
                    await window.viewer.importXML(xml);
                    const canvas = window.viewer.get('canvas');
                    canvas.zoom('fit-viewport'); 
                    const viewbox = canvas.viewbox();
                    return {
                        scale: viewbox.scale,
                        x_offset: viewbox.x,
                        y_offset: viewbox.y,
                        success: true
                    };
                } catch (err) {
                    return { success: false, error: err.message };
                }
            };
        </script>
    </body>
    </html>
    """

    # Сохраняем актуальный список классов для YOLO
    with open(os.path.join(folder_path, 'classes.txt'), 'w') as f:
        f.write('\n'.join(CLASSES))

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        img_w, img_h = 1600, 1600
        await page.set_viewport_size({"width": img_w, "height": img_h})
        await page.set_content(html_template)

        ns = {
            'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
            'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
            'omgdc': 'http://www.omg.org/spec/DD/20100524/DC'
        }

        for file in os.listdir(folder_path):
            if not file.endswith(".bpmn"): continue

            xml_path = os.path.join(folder_path, file)
            base_name = os.path.splitext(file)[0]
            png_path = os.path.join(folder_path, base_name + ".png")
            txt_path = os.path.join(folder_path, base_name + ".txt")

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                with open(xml_path, 'r', encoding='utf-8') as f:
                    xml_content = f.read()

                render_meta = await page.evaluate("xml => window.getRenderData(xml)", xml_content)
                if not render_meta['success']:
                    print(f"⚠️ Ошибка рендеринга {file}: {render_meta.get('error')}")
                    continue

                await asyncio.sleep(0.6)
                await page.screenshot(path=png_path)

                scale, off_x, off_y = render_meta['scale'], render_meta['x_offset'], render_meta['y_offset']
                yolo_labels = []

                element_map = {}
                for el in root.iter():
                    tag = el.tag.split('}')[-1]
                    el_id = el.get('id')
                    if el_id:
                        element_map[el_id] = tag

                for shape in root.findall('.//bpmndi:BPMNShape', ns):
                    bpmn_id = shape.get('bpmnElement')
                    tag = element_map.get(bpmn_id, "")

                    # Игнорируем контейнеры и связи, чтобы модель не путалась
                    if tag in ['participant', 'lane', 'collaboration', 'textAnnotation', 'association']:
                        continue

                    bounds = shape.find('omgdc:Bounds', ns)
                    if bounds is None: continue

                    cls = -1
                    tag_l = tag.lower()

                    # 0: Tasks, Call Activities, Subprocesses (Прямоугольники)
                    if 'task' in tag_l or 'callactivity' in tag_l or 'subprocess' in tag_l:
                        cls = 0
                    # 1: Gateways (Ромбы)
                    elif 'gateway' in tag_l:
                        cls = 1
                    # 2: Start Events (Тонкий круг)
                    elif 'startevent' in tag_l:
                        cls = 2
                    # 3: End Events (Жирный круг)
                    elif 'endevent' in tag_l:
                        cls = 3
                    # 4: Intermediate / Boundary Events (Двойной круг)
                    elif 'intermediate' in tag_l or 'boundaryevent' in tag_l:
                        cls = 4

                    if cls != -1:
                        x = (float(bounds.get('x')) - off_x) * scale
                        y = (float(bounds.get('y')) - off_y) * scale
                        w = float(bounds.get('width')) * scale
                        h = float(bounds.get('height')) * scale

                        cx = (x + w / 2) / img_w
                        cy = (y + h / 2) / img_h
                        nw, nh = w / img_w, h / img_h

                        if 0 <= cx <= 1 and 0 <= cy <= 1:
                            yolo_labels.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                with open(txt_path, 'w') as f:
                    f.write("\n".join(yolo_labels))
                print(f"✅ {file}: записано {len(yolo_labels)} узлов")

            except Exception as e:
                print(f"❌ Ошибка {file}: {e}")

        await browser.close()


if __name__ == "__main__":
    # Убедись, что путь правильный
    path = r'C:\Users\VelmorSDFG\PycharmProjects\BPMN\uploads\raw\bpmn\02-Results'
    asyncio.run(process_bpmn_dataset(path))