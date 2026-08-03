let currentGraphOrders = [];
let orderAIndex = 0;
let orderBIndex = 0;
let currentZoomPercent = 100;

const graphSelect = document.getElementById('graphSelect');
const vizContainer = document.getElementById('vizContainer');
const nav = document.getElementById('navigation');
const graphStatsPanel = document.getElementById('graphStatsPanel');

// Inspector DOM Panel Elements
const detailsPanel = document.getElementById('detailsPanel');
const inspectOp = document.getElementById('inspectOp');
const inspectName = document.getElementById('inspectName');
const inspectEngine = document.getElementById('inspectEngine');
const inspectInterval = document.getElementById('inspectInterval');
const inspectDuration = document.getElementById('inspectDuration');

// Dropdowns for Orders
const selectOrderA = document.getElementById('selectOrderA');
const selectOrderB = document.getElementById('selectOrderB');

// Pager buttons
const prevBtnA = document.getElementById('prevBtnA');
const nextBtnA = document.getElementById('nextBtnA');
const prevBtnB = document.getElementById('prevBtnB');
const nextBtnB = document.getElementById('nextBtnB');

// Zoom controls
const zoomRange = document.getElementById('zoomRange');
const zoomVal = document.getElementById('zoomVal');
const zoomResetBtn = document.getElementById('zoomResetBtn');

// Known OP colors palette
const KNOWN_OP_COLORS = {
    'INPUT': { bg: 'var(--color-input-bg)', text: 'var(--color-input-text)', border: 'var(--color-input-border)' },
    'ADD': { bg: 'var(--color-add-bg)', text: 'var(--color-add-text)', border: 'var(--color-add-border)' },
    'MUL': { bg: 'var(--color-mul-bg)', text: 'var(--color-mul-text)', border: 'var(--color-mul-border)' },
    'COPYTO': { bg: 'var(--color-copyto-bg)', text: 'var(--color-copyto-text)', border: 'var(--color-copyto-border)' },
    'COPY_TO': { bg: 'var(--color-copyto-bg)', text: 'var(--color-copyto-text)', border: 'var(--color-copyto-border)' },
    'SQRT': { bg: 'var(--color-sqrt-bg)', text: 'var(--color-sqrt-text)', border: 'var(--color-sqrt-border)' },
    'DOT': { bg: '#f0fdf4', text: '#15803d', border: '#bbf7d0' },
    'CONCAT': { bg: '#fff7ed', text: '#c2410c', border: '#fed7aa' },
    'RESHAPE': { bg: '#f0f9ff', text: '#0369a1', border: '#bae6fd' },
    'PERMUTE': { bg: '#f5f3ff', text: '#7c3aed', border: '#ddd6fe' },
    'SLICE': { bg: '#fef2f2', text: '#b91c1c', border: '#fecaca' },
    'GATHER': { bg: '#ecfdf5', text: '#047857', border: '#a7f3d0' },
    'CAST': { bg: '#fdf4ff', text: '#a21caf', border: '#f5d0fe' },
    'MAX': { bg: '#fef1f2', text: '#be123c', border: '#ffe4e6' },
    'SUM': { bg: '#f0fdfa', text: '#0f766e', border: '#99f6e4' }
};

function getOpColor(op) {
    const cleanOp = op ? op.replace('Op.', '').trim() : 'UNKNOWN';
    if (KNOWN_OP_COLORS[cleanOp]) {
        return KNOWN_OP_COLORS[cleanOp];
    }
    let hash = 0;
    for (let i = 0; i < cleanOp.length; i++) {
        hash = cleanOp.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    return {
        bg: `hsl(${hue}, 80%, 96%)`,
        text: `hsl(${hue}, 70%, 30%)`,
        border: `hsl(${hue}, 60%, 80%)`
    };
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

function renderLegend(orders) {
    const legendContainer = document.querySelector('.legend');
    if (!legendContainer) return;

    const uniqueOps = new Set();
    orders.forEach(orderObj => {
        orderObj.schedule.forEach(task => uniqueOps.add(task.op));
    });

    legendContainer.innerHTML = '';
    Array.from(uniqueOps).sort().forEach(op => {
        const color = getOpColor(op);
        const item = document.createElement('span');
        item.className = 'legend-item';

        const dot = document.createElement('span');
        dot.className = 'legend-dot';
        dot.style.backgroundColor = color.text;

        item.appendChild(dot);
        item.appendChild(document.createTextNode(` ${op}`));
        legendContainer.appendChild(item);
    });
}

/**
 * Handles fetching graph payload and setting up application states.
 */
async function loadGraphData(name) {
    if (!name) return;
    try {
        const resp = await fetch(`/api/graph/${encodeURIComponent(name)}`);
        const data = await resp.json();

        currentGraphOrders = data.orders.map(orderObj => {
            const parsedNodes = orderObj.ordered.map(item => typeof item === 'string' ? JSON.parse(item) : item);
            const parsedBuffers = orderObj.buffers.map(item => typeof item === 'string' ? JSON.parse(item) : item);
            const parsedAllocated = orderObj.allocated.map(item => typeof item === 'string' ? JSON.parse(item) : item);

            const bufferIdxToNode = {};
            let bufferCount = 0;
            parsedNodes.forEach(node => {
                const isStorage = node.mem_space &&
                    node.mem_space.idx === 0 &&
                    (node.mem_space.handle_type === 'Handle.STORAGE' || node.mem_space.handle_type === 0);
                if (isStorage) {
                    return;
                }
                bufferIdxToNode[bufferCount] = node;
                bufferCount++;
            });

            // Map parsed nodes to scheduled tasks
            const schedule = parsedNodes.map(node => {
                const opName = node.op ? node.op.replace('Op.', '') : 'INPUT';
                const engineKey = typeof node.engine === 'string'
                    ? node.engine
                    : (node.engine ? `Engine(idx=${node.engine.idx}, engine_type=${node.engine.engine_type})` : 'Engine(idx=0, engine_type=EngineType.CPU)');
                const duration = node.duration !== undefined ? node.duration : (node.cost !== undefined ? node.cost : 1);
                const start = node.start !== undefined ? node.start : node.birth;

                return {
                    name: node.name,
                    op: opName,
                    start: start,
                    end: start + duration,
                    duration: duration,
                    engine: engineKey,
                    size: node.size
                };
            });

            // Map parsed buffers
            const buffers = parsedBuffers.map(buf => {
                const node = bufferIdxToNode[buf.idx];
                const nodeName = buf.node_name || (node ? node.name : `Buf ${buf.idx}`);
                const opName = buf.op ? buf.op.replace('Op.', '') : (node && node.op ? node.op.replace('Op.', '') : 'INPUT');
                const memSpaceIdx = buf.mem_space_idx !== undefined ? buf.mem_space_idx : (buf.mem_space ? buf.mem_space.idx : 1);
                const memSpaceHandle = buf.mem_space_handle || (buf.mem_space && buf.mem_space.handle_type ? String(buf.mem_space.handle_type).replace('Handle.', '') : 'CPP');

                return {
                    idx: buf.idx,
                    node_name: nodeName,
                    op: opName,
                    start: buf.start,
                    end: buf.end,
                    offset: buf.offset,
                    size: buf.size,
                    mem_space_idx: memSpaceIdx,
                    mem_space_handle: memSpaceHandle
                };
            });

            // Map parsed allocated buffers
            const allocated = parsedAllocated.map(buf => {
                const node = bufferIdxToNode[buf.idx];
                const nodeName = buf.node_name || (node ? node.name : `Buf ${buf.idx}`);
                const opName = buf.op ? buf.op.replace('Op.', '') : (node && node.op ? node.op.replace('Op.', '') : 'INPUT');
                const memSpaceIdx = buf.mem_space_idx !== undefined ? buf.mem_space_idx : (buf.mem_space ? buf.mem_space.idx : 1);
                const memSpaceHandle = buf.mem_space_handle || (buf.mem_space && buf.mem_space.handle_type ? String(buf.mem_space.handle_type).replace('Handle.', '') : 'CPP');

                return {
                    idx: buf.idx,
                    node_name: nodeName,
                    op: opName,
                    start: buf.start,
                    end: buf.end,
                    offset: buf.offset,
                    size: buf.size,
                    mem_space_idx: memSpaceIdx,
                    mem_space_handle: memSpaceHandle
                };
            });

            return {
                schedule: schedule,
                buffers: buffers,
                allocated: allocated
            };
        });

        graphStatsPanel.classList.remove('hidden');
        nav.classList.remove('hidden');

        renderLegend(currentGraphOrders);
        calculateStats();
        renderComparison();
    } catch (e) {
        console.error("Error fetching execution schedules: ", e);
    }
}

// Event Listeners
graphSelect.addEventListener('change', (e) => loadGraphData(e.target.value));

selectOrderA.addEventListener('change', (e) => {
    orderAIndex = parseInt(e.target.value);
    updateCostBadges();
    renderComparison();
});

selectOrderB.addEventListener('change', (e) => {
    orderBIndex = parseInt(e.target.value);
    updateCostBadges();
    renderComparison();
});

prevBtnA.addEventListener('click', () => {
    if (orderAIndex > 0) {
        orderAIndex--;
        selectOrderA.value = orderAIndex;
        updateCostBadges();
        renderComparison();
    }
});

nextBtnA.addEventListener('click', () => {
    if (orderAIndex < currentGraphOrders.length - 1) {
        orderAIndex++;
        selectOrderA.value = orderAIndex;
        updateCostBadges();
        renderComparison();
    }
});

prevBtnB.addEventListener('click', () => {
    if (orderBIndex > 0) {
        orderBIndex--;
        selectOrderB.value = orderBIndex;
        updateCostBadges();
        renderComparison();
    }
});

nextBtnB.addEventListener('click', () => {
    if (orderBIndex < currentGraphOrders.length - 1) {
        orderBIndex++;
        selectOrderB.value = orderBIndex;
        updateCostBadges();
        renderComparison();
    }
});

// Zoom Event Listeners
if (zoomRange) {
    zoomRange.addEventListener('input', (e) => {
        currentZoomPercent = parseInt(e.target.value, 10);
        if (zoomVal) zoomVal.textContent = `${currentZoomPercent}%`;
        renderComparison();
    });
}

if (zoomResetBtn) {
    zoomResetBtn.addEventListener('click', () => {
        currentZoomPercent = 100;
        if (zoomRange) zoomRange.value = 100;
        if (zoomVal) zoomVal.textContent = '100%';
        renderComparison();
    });
}

function formatEngineName(rawName) {
    if (typeof rawName !== 'string') return { name: String(rawName), sub: '' };
    const idxMatch = rawName.match(/idx=(\d+)/);
    const typeMatch = rawName.match(/engine_type=(?:EngineType\.)?(\w+)/i) || rawName.match(/EngineType\.(\w+)/i);
    if (idxMatch && typeMatch) {
        const idx = idxMatch[1];
        const type = typeMatch[1].replace('_', ' ');
        return {
            name: `${type}`,
            sub: `ID: ${idx}`
        };
    }
    return { name: rawName, sub: '' };
}

function calculateStats() {
    if (!currentGraphOrders || currentGraphOrders.length === 0) return;

    let minCost = Infinity;
    let maxCost = -Infinity;
    let minIdx = 0;
    let maxIdx = 0;

    currentGraphOrders.forEach((orderObj, idx) => {
        const schedule = orderObj.schedule;
        const cost = schedule.length > 0 ? Math.max(...schedule.map(t => t.end)) : 0;
        if (cost < minCost) {
            minCost = cost;
            minIdx = idx;
        }
        if (cost > maxCost) {
            maxCost = cost;
            maxIdx = idx;
        }
    });

    const statTotalOrders = document.getElementById('statTotalOrders');
    const statBestOrder = document.getElementById('statBestOrder');
    const statWorstOrder = document.getElementById('statWorstOrder');

    statTotalOrders.textContent = currentGraphOrders.length;
    statBestOrder.textContent = `Order ${minIdx + 1} (Cost: ${minCost.toFixed(2)})`;
    statWorstOrder.textContent = `Order ${maxIdx + 1} (Cost: ${maxCost.toFixed(2)})`;

    orderAIndex = minIdx;
    orderBIndex = maxIdx;

    populateDropdowns(minIdx, maxIdx);
}

function populateDropdowns(defaultA, defaultB) {
    selectOrderA.innerHTML = '';
    selectOrderB.innerHTML = '';

    currentGraphOrders.forEach((orderObj, idx) => {
        const schedule = orderObj.schedule;
        const cost = schedule.length > 0 ? Math.max(...schedule.map(t => t.end)) : 0;

        const optionA = document.createElement('option');
        optionA.value = idx;
        optionA.textContent = `Order ${idx + 1} (Cost: ${cost.toFixed(2)})`;
        selectOrderA.appendChild(optionA);

        const optionB = document.createElement('option');
        optionB.value = idx;
        optionB.textContent = `Order ${idx + 1} (Cost: ${cost.toFixed(2)})`;
        selectOrderB.appendChild(optionB);
    });

    selectOrderA.value = defaultA;
    selectOrderB.value = defaultB;

    updateCostBadges();
}

function updateCostBadges() {
    const costAElement = document.getElementById('totalCostA');
    const costBElement = document.getElementById('totalCostB');

    const orderAObj = currentGraphOrders[orderAIndex];
    const orderBObj = currentGraphOrders[orderBIndex];

    const scheduleA = orderAObj ? orderAObj.schedule : [];
    const scheduleB = orderBObj ? orderBObj.schedule : [];

    const costA = scheduleA && scheduleA.length > 0 ? Math.max(...scheduleA.map(t => t.end)) : 0;
    const costB = scheduleB && scheduleB.length > 0 ? Math.max(...scheduleB.map(t => t.end)) : 0;

    costAElement.textContent = costA.toFixed(2);
    costBElement.textContent = costB.toFixed(2);
}

function updateButtonStates() {
    prevBtnA.disabled = (orderAIndex === 0);
    nextBtnA.disabled = (orderAIndex === currentGraphOrders.length - 1);
    prevBtnB.disabled = (orderBIndex === 0);
    nextBtnB.disabled = (orderBIndex === currentGraphOrders.length - 1);
}

function renderComparison() {
    vizContainer.innerHTML = '';

    const orderAObj = currentGraphOrders[orderAIndex];
    const orderBObj = currentGraphOrders[orderBIndex];

    if (!orderAObj || !orderBObj) return;

    updateButtonStates();

    const scheduleA = orderAObj.schedule;
    const scheduleB = orderBObj.schedule;
    const buffersA = orderAObj.buffers;
    const buffersB = orderBObj.buffers;
    const allocatedA = orderAObj.allocated;
    const allocatedB = orderBObj.allocated;

    const maxTimeA = scheduleA.length > 0 ? Math.max(...scheduleA.map(t => t.end)) : 0;
    const maxTimeB = scheduleB.length > 0 ? Math.max(...scheduleB.map(t => t.end)) : 0;
    const commonMaxTime = Math.max(maxTimeA, maxTimeB, 1);

    // Adaptive pixel scale factor multiplied by horizontal zoom factor
    const baseScale = commonMaxTime > 0 ? Math.max(10, Math.min(100, 1200 / commonMaxTime)) : 50;
    const scale = baseScale * (currentZoomPercent / 100);

    const blockA = document.createElement('div');
    blockA.className = 'schedule-block schedule-block-a';
    renderScheduleInto(blockA, scheduleA, buffersA, allocatedA, orderAIndex, commonMaxTime, scale, 'A');
    vizContainer.appendChild(blockA);

    const divider = document.createElement('div');
    divider.className = 'schedule-divider';
    vizContainer.appendChild(divider);

    const blockB = document.createElement('div');
    blockB.className = 'schedule-block schedule-block-b';
    renderScheduleInto(blockB, scheduleB, buffersB, allocatedB, orderBIndex, commonMaxTime, scale, 'B');
    vizContainer.appendChild(blockB);
}

function renderScheduleInto(container, order, buffers, allocated, orderIndex, maxTime, scale, blockLabel) {
    const timelineWidthStr = (maxTime * scale) + 'px';

    const headerRow = document.createElement('div');
    headerRow.className = 'schedule-block-header';

    const tag = document.createElement('span');
    tag.className = `schedule-block-tag tag-${blockLabel.toLowerCase()}`;
    tag.textContent = `Schedule ${blockLabel}`;
    headerRow.appendChild(tag);

    const cost = order.length > 0 ? Math.max(...order.map(t => t.end)) : 0;
    const labelSpan = document.createElement('span');
    labelSpan.innerHTML = ` &mdash; Order Index: <strong>${orderIndex + 1}</strong> | Total Cost: `;
    headerRow.appendChild(labelSpan);

    const costValue = document.createElement('span');
    costValue.className = `order-cost-val cost-${blockLabel.toLowerCase()}`;
    costValue.textContent = cost.toFixed(2);
    headerRow.appendChild(costValue);

    container.appendChild(headerRow);

    const blockBody = document.createElement('div');
    blockBody.className = 'schedule-block-body';

    if (order.length === 0) {
        const emptyMsg = document.createElement('div');
        emptyMsg.className = 'empty-schedule-msg';
        emptyMsg.textContent = 'This schedule does not contain any tasks.';
        blockBody.appendChild(emptyMsg);
        container.appendChild(blockBody);
        return;
    }

    const engines = [...new Set(order.map(t => t.engine))].sort();

    // 1. Time Ruler Row
    const rulerRow = document.createElement('div');
    rulerRow.className = 'ruler-row';

    const rulerSpacer = document.createElement('div');
    rulerSpacer.className = 'ruler-label-spacer';
    rulerRow.appendChild(rulerSpacer);

    const rulerTicks = document.createElement('div');
    rulerTicks.className = 'ruler-ticks';
    rulerTicks.style.width = timelineWidthStr;
    rulerTicks.style.flexShrink = '0';

    const tickInterval = maxTime > 100 ? Math.ceil(maxTime / 20) : (maxTime > 30 ? 5 : (maxTime > 15 ? 2 : 1));

    for (let t = 0; t <= maxTime; t += tickInterval) {
        const tick = document.createElement('div');
        tick.className = 'ruler-tick';
        tick.style.left = (t * scale) + 'px';
        tick.textContent = Number.isInteger(t) ? t : t.toFixed(1);
        rulerTicks.appendChild(tick);
    }
    rulerRow.appendChild(rulerTicks);
    blockBody.appendChild(rulerRow);

    // 2. Grid Overlay
    const gridOverlay = document.createElement('div');
    gridOverlay.className = 'time-grid-overlay';

    for (let t = 0; t <= maxTime; t += tickInterval) {
        const gridLine = document.createElement('div');
        gridLine.style.position = 'absolute';
        gridLine.style.left = (t * scale) + 'px';
        gridLine.style.top = '0';
        gridLine.style.bottom = '0';
        gridLine.style.width = '1px';
        gridLine.style.backgroundColor = '#e2e8f0';
        gridLine.style.zIndex = '1';
        gridOverlay.appendChild(gridLine);
    }
    blockBody.appendChild(gridOverlay);

    // 3. Hardware Track Rows
    engines.forEach(engName => {
        const row = document.createElement('div');
        row.className = 'engine-row';

        const label = document.createElement('div');
        label.className = 'engine-label';

        const formatted = formatEngineName(engName);
        const nameSpan = document.createElement('span');
        nameSpan.textContent = formatted.name;
        const subSpan = document.createElement('span');
        subSpan.className = 'engine-type-badge';
        subSpan.textContent = formatted.sub;

        label.appendChild(nameSpan);
        label.appendChild(subSpan);

        const timeline = document.createElement('div');
        timeline.className = 'timeline';
        timeline.style.width = timelineWidthStr;
        timeline.style.flexShrink = '0';

        const engineTasks = order.filter(t => t.engine === engName);
        engineTasks.forEach(task => {
            const bar = document.createElement('div');

            const isZero = task.duration === 0;
            const barWidth = isZero ? 24 : (task.duration * scale);
            const offsetLeft = task.start * scale;

            const color = getOpColor(task.op);

            bar.className = 'task-bar';
            if (isZero) bar.classList.add('task-zero-duration');

            bar.style.left = offsetLeft + 'px';
            bar.style.width = barWidth + 'px';
            bar.style.backgroundColor = color.bg;
            bar.style.color = color.text;
            bar.style.borderColor = color.border;

            bar.textContent = task.name;
            bar.title = `${task.name} (${task.op}): Time ${task.start}-${task.end} (duration: ${task.duration})`;

            bar.addEventListener('mouseenter', () => showInspector(task, formatted));
            bar.addEventListener('mouseleave', () => hideInspector());

            timeline.appendChild(bar);
        });

        row.appendChild(label);
        row.appendChild(timeline);
        blockBody.appendChild(row);
    });

    // 4. Logical Memory Tracks
    if (buffers && buffers.length > 0) {
        const memSectionHeader = document.createElement('div');
        memSectionHeader.className = 'mem-section-header';
        memSectionHeader.innerHTML = '<span>Logical Buffer Lifetimes (Active Spans)</span>';
        blockBody.appendChild(memSectionHeader);

        const memSpaces = [...new Set(buffers.map(b => b.mem_space_idx))].sort();

        memSpaces.forEach(memIdx => {
            const spaceBuffers = buffers.filter(b => b.mem_space_idx === memIdx);

            const row = document.createElement('div');
            row.className = 'mem-row';

            const label = document.createElement('div');
            label.className = 'mem-label';

            const spaceName = memIdx === 1 ? 'CPU Memory' : (memIdx === 2 ? 'GPU Memory' : `Mem Space ${memIdx}`);
            const handleName = spaceBuffers.length > 0 ? spaceBuffers[0].mem_space_handle : '';

            const nameSpan = document.createElement('span');
            nameSpan.textContent = spaceName;
            const subSpan = document.createElement('span');
            subSpan.className = 'mem-type-badge';
            subSpan.textContent = `Handle: ${handleName}`;

            label.appendChild(nameSpan);
            label.appendChild(subSpan);

            const memTimeline = document.createElement('div');
            memTimeline.className = 'mem-timeline';
            memTimeline.style.width = timelineWidthStr;
            memTimeline.style.flexShrink = '0';

            const numBufs = spaceBuffers.length;
            spaceBuffers.forEach((buf, bIdx) => {
                const bar = document.createElement('div');
                const color = getOpColor(buf.op);
                bar.className = 'mem-buffer-bar';

                const barWidth = (buf.end - buf.start) * scale;
                const offsetLeft = buf.start * scale;

                bar.style.left = offsetLeft + 'px';
                bar.style.width = barWidth + 'px';
                bar.style.backgroundColor = color.bg;
                bar.style.color = color.text;
                bar.style.borderColor = color.border;

                const pctHeight = 100 / numBufs;
                const pctBottom = bIdx * pctHeight;

                bar.style.height = `calc(${pctHeight}% - 4px)`;
                bar.style.bottom = `calc(${pctBottom}% + 2px)`;

                bar.innerHTML = `<span class="mem-buffer-label">${buf.node_name}</span>`;
                bar.title = `Buffer ${buf.idx} (${buf.node_name}): Active ${buf.start}-${buf.end}, Size: ${formatBytes(buf.size)}`;

                bar.addEventListener('mouseenter', () => showMemInspector(buf, spaceName, handleName));
                bar.addEventListener('mouseleave', () => hideInspector());

                memTimeline.appendChild(bar);
            });

            row.appendChild(label);
            row.appendChild(memTimeline);
            blockBody.appendChild(row);
        });
    }

    // 5. Physical Memory Layout (Allocated Offsets)
    const physSectionHeader = document.createElement('div');
    physSectionHeader.className = 'mem-section-header';
    physSectionHeader.innerHTML = '<span>Physical Memory Allocation Layout (malloc offsets)</span>';
    blockBody.appendChild(physSectionHeader);

    if (allocated && allocated.length > 0) {
        const memSpaces = [...new Set(allocated.map(b => b.mem_space_idx))].sort();

        memSpaces.forEach(memIdx => {
            const spaceBuffers = allocated.filter(b => b.mem_space_idx === memIdx);

            const row = document.createElement('div');
            row.className = 'mem-row';

            const label = document.createElement('div');
            label.className = 'mem-label';

            const spaceName = memIdx === 1 ? 'CPU Memory' : (memIdx === 2 ? 'GPU Memory' : `Mem Space ${memIdx}`);
            const handleName = spaceBuffers.length > 0 ? spaceBuffers[0].mem_space_handle : '';

            const nameSpan = document.createElement('span');
            nameSpan.textContent = spaceName;
            const subSpan = document.createElement('span');
            subSpan.className = 'mem-type-badge';
            subSpan.textContent = `Handle: ${handleName}`;

            label.appendChild(nameSpan);
            label.appendChild(subSpan);

            const memTimeline = document.createElement('div');
            memTimeline.className = 'mem-timeline';
            memTimeline.style.width = timelineWidthStr;
            memTimeline.style.flexShrink = '0';

            const maxOffsetAndSize = Math.max(...spaceBuffers.map(b => (b.offset >= 0 ? b.offset : 0) + b.size), 1);

            // Bounded Y-axis guideline gridlines
            const numYGridlines = 4;
            const yStep = maxOffsetAndSize / numYGridlines;
            for (let i = 0; i <= numYGridlines; i++) {
                const y = Math.round(i * yStep);
                const line = document.createElement('div');
                line.className = 'mem-offset-gridline';
                line.style.bottom = `${(y / maxOffsetAndSize) * 100}%`;
                memTimeline.appendChild(line);

                if (i < numYGridlines) {
                    const tickText = document.createElement('span');
                    tickText.className = 'mem-offset-tick-text';
                    tickText.textContent = `O: ${formatBytes(y)}`;
                    tickText.style.bottom = `${(y / maxOffsetAndSize) * 100 + 1}%`;
                    memTimeline.appendChild(tickText);
                }
            }

            spaceBuffers.forEach(buf => {
                const bar = document.createElement('div');
                const color = getOpColor(buf.op);
                bar.className = 'mem-buffer-bar';

                const barWidth = (buf.end - buf.start) * scale;
                const offsetLeft = buf.start * scale;

                bar.style.left = offsetLeft + 'px';
                bar.style.width = barWidth + 'px';
                bar.style.backgroundColor = color.bg;
                bar.style.color = color.text;
                bar.style.borderColor = color.border;

                const bufOffset = buf.offset >= 0 ? buf.offset : 0;
                const pctHeight = (buf.size / maxOffsetAndSize) * 100;
                const pctBottom = (bufOffset / maxOffsetAndSize) * 100;

                bar.style.height = `calc(${pctHeight}% - 4px)`;
                bar.style.bottom = `calc(${pctBottom}% + 2px)`;

                bar.innerHTML = `<span class="mem-buffer-label">${buf.node_name}</span>`;
                bar.title = `Buffer ${buf.idx} (${buf.node_name}): Active ${buf.start}-${buf.end}, Offset: ${formatBytes(bufOffset)}, Size: ${formatBytes(buf.size)}`;

                bar.addEventListener('mouseenter', () => showMemInspector(buf, spaceName, handleName));
                bar.addEventListener('mouseleave', () => hideInspector());

                memTimeline.appendChild(bar);
            });

            row.appendChild(label);
            row.appendChild(memTimeline);
            blockBody.appendChild(row);
        });
    } else {
        const emptyMsg = document.createElement('div');
        emptyMsg.className = 'empty-schedule-msg';
        emptyMsg.textContent = 'No memory layout was successfully allocated (Allocated 0 buffers).';
        blockBody.appendChild(emptyMsg);
    }

    container.appendChild(blockBody);
}

function showInspector(task, engineDetails) {
    detailsPanel.classList.remove('hidden');

    const color = getOpColor(task.op);
    inspectOp.className = 'inspect-badge';
    inspectOp.style.backgroundColor = color.bg;
    inspectOp.style.color = color.text;
    inspectOp.style.borderColor = color.border;
    inspectOp.textContent = task.op;

    inspectName.textContent = `Node: ${task.name}`;
    inspectEngine.textContent = `${engineDetails.name} (${engineDetails.sub})`;
    inspectInterval.textContent = `${task.start} → ${task.end}`;
    inspectDuration.textContent = `${task.duration} unit${task.duration !== 1 ? 's' : ''}`;
}

function showMemInspector(buf, spaceName, handleName) {
    detailsPanel.classList.remove('hidden');

    const color = getOpColor(buf.op);
    inspectOp.className = 'inspect-badge';
    inspectOp.style.backgroundColor = color.bg;
    inspectOp.style.color = color.text;
    inspectOp.style.borderColor = color.border;
    inspectOp.textContent = buf.op;

    inspectName.textContent = `Buffer: ${buf.node_name}`;
    inspectEngine.textContent = `${spaceName} (${handleName})`;
    inspectInterval.textContent = `Active: ${buf.start} → ${buf.end}`;
    inspectDuration.textContent = `Offset: ${formatBytes(buf.offset >= 0 ? buf.offset : 0)} | Size: ${formatBytes(buf.size)}`;
}

function hideInspector() {
    // Keep last hovered item details displayed
}

if (graphSelect && graphSelect.value) {
    loadGraphData(graphSelect.value);
}