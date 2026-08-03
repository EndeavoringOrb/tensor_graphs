let currentGraphOrders = [];
let orderAIndex = 0;
let orderBIndex = 0;
let currentZoomPercent = 100;

// Track renderers registered for viewport canvas rendering
let activeTrackRenderers = [];
let renderScheduled = false;

const graphSelect = document.getElementById('graphSelect');
const vizContainer = document.getElementById('vizContainer');
const nav = document.getElementById('navigation');
const graphStatsPanel = document.getElementById('graphStatsPanel');
const scrollContainer = document.querySelector('.timeline-container-scroll');

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

// Known OP colors palette (Hex format for high performance Canvas fills)
const KNOWN_OP_COLORS = {
    'INPUT': { bg: '#f1f5f9', text: '#475569', border: '#cbd5e1' },
    'ADD': { bg: '#eff6ff', text: '#1d4ed8', border: '#bfdbfe' },
    'MUL': { bg: '#faf5ff', text: '#6d28d9', border: '#e9d5ff' },
    'COPYTO': { bg: '#fffbeb', text: '#b45309', border: '#fde047' },
    'COPY_TO': { bg: '#fffbeb', text: '#b45309', border: '#fde047' },
    'SQRT': { bg: '#fdf2f8', text: '#be185d', border: '#fbcfe8' },
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
 * Binary search to find starting index in items sorted by start time.
 */
function findStartIndex(items, targetTime) {
    let low = 0;
    let high = items.length - 1;
    let ans = 0;
    while (low <= high) {
        const mid = (low + high) >> 1;
        if (items[mid].end >= targetTime) {
            ans = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return ans;
}

/**
 * Dynamic Tick interval calculation for high-performance scale markers.
 */
function getNiceTickInterval(rawInterval) {
    if (rawInterval <= 0) return 1;
    const exponent = Math.floor(Math.log10(rawInterval));
    const fraction = rawInterval / Math.pow(10, exponent);
    let niceFraction;
    if (fraction <= 1.5) niceFraction = 1;
    else if (fraction <= 3) niceFraction = 2;
    else if (fraction <= 7) niceFraction = 5;
    else niceFraction = 10;
    return niceFraction * Math.pow(10, exponent);
}

/**
 * Canvas drawing helpers.
 */
function drawRoundedRect(ctx, x, y, width, height, radius) {
    if (width < 2 * radius) radius = width / 2;
    if (height < 2 * radius) radius = height / 2;
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
}

function getTruncatedText(ctx, text, maxWidth) {
    if (maxWidth <= 10) return '';
    if (ctx.measureText(text).width <= maxWidth) return text;
    let ellipsis = '…';
    let truncated = text;
    while (truncated.length > 0 && ctx.measureText(truncated + ellipsis).width > maxWidth) {
        truncated = truncated.substring(0, truncated.length - 1);
    }
    return truncated ? truncated + ellipsis : '';
}

/**
 * Fetches graph payload and parses ordered nodes into pre-sorted arrays.
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
            }).sort((a, b) => a.start - b.start);

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
            }).sort((a, b) => a.start - b.start);

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
            }).sort((a, b) => a.start - b.start);

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

if (scrollContainer) {
    scrollContainer.addEventListener('scroll', () => {
        requestViewportRender();
    }, { passive: true });
}

window.addEventListener('resize', () => {
    requestViewportRender();
});

function requestViewportRender() {
    if (!renderScheduled) {
        renderScheduled = true;
        window.requestAnimationFrame(() => {
            renderVisibleTrackViewports();
            renderScheduled = false;
        });
    }
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
    activeTrackRenderers = [];

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

    requestViewportRender();
}

function renderScheduleInto(container, order, buffers, allocated, orderIndex, maxTime, scale, blockLabel) {
    const timelineWidthPx = Math.ceil(maxTime * scale);
    const timelineWidthStr = timelineWidthPx + 'px';

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

    rulerRow.appendChild(rulerTicks);
    blockBody.appendChild(rulerRow);

    registerTrackRenderer({
        type: 'ruler',
        container: rulerTicks,
        maxTime: maxTime,
        scale: scale
    });

    // 2. Hardware Track Rows
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

        const canvas = document.createElement('canvas');
        canvas.className = 'track-canvas';
        timeline.appendChild(canvas);

        const engineTasks = order.filter(t => t.engine === engName);

        const renderer = {
            type: 'engine',
            container: timeline,
            canvas: canvas,
            ctx: canvas.getContext('2d'),
            data: engineTasks,
            scale: scale,
            rowHeight: 72,
            extra: { formatted: formatted },
            hoveredItem: null
        };

        setupHoverInteractivity(renderer);
        registerTrackRenderer(renderer);

        row.appendChild(label);
        row.appendChild(timeline);
        blockBody.appendChild(row);
    });

    // 3. Logical Memory Tracks
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

            const canvas = document.createElement('canvas');
            canvas.className = 'track-canvas';
            memTimeline.appendChild(canvas);

            const renderer = {
                type: 'logical',
                container: memTimeline,
                canvas: canvas,
                ctx: canvas.getContext('2d'),
                data: spaceBuffers,
                scale: scale,
                rowHeight: 110,
                extra: { spaceName, handleName, numBufs: spaceBuffers.length },
                hoveredItem: null
            };

            setupHoverInteractivity(renderer);
            registerTrackRenderer(renderer);

            row.appendChild(label);
            row.appendChild(memTimeline);
            blockBody.appendChild(row);
        });
    }

    // 4. Physical Memory Layout (Allocated Offsets)
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

            const canvas = document.createElement('canvas');
            canvas.className = 'track-canvas';
            memTimeline.appendChild(canvas);

            const maxOffsetAndSize = Math.max(...spaceBuffers.map(b => (b.offset >= 0 ? b.offset : 0) + b.size), 1);

            const renderer = {
                type: 'physical',
                container: memTimeline,
                canvas: canvas,
                ctx: canvas.getContext('2d'),
                data: spaceBuffers,
                scale: scale,
                rowHeight: 110,
                extra: { spaceName, handleName, maxOffsetAndSize },
                hoveredItem: null
            };

            setupHoverInteractivity(renderer);
            registerTrackRenderer(renderer);

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

function registerTrackRenderer(renderer) {
    activeTrackRenderers.push(renderer);
}

/**
 * Main viewport view update pass. Executes fast binary range rendering on canvas.
 */
function renderVisibleTrackViewports() {
    if (!scrollContainer) return;

    const scrollLeft = scrollContainer.scrollLeft;
    const viewportWidth = scrollContainer.clientWidth || 1200;
    const dpr = window.devicePixelRatio || 1;

    activeTrackRenderers.forEach(track => {
        if (track.type === 'ruler') {
            renderRulerTicks(track, scrollLeft, viewportWidth);
            return;
        }

        const canvas = track.canvas;
        const ctx = track.ctx;
        const scale = track.scale;
        const rowHeight = track.rowHeight;

        canvas.style.left = scrollLeft + 'px';
        canvas.style.width = viewportWidth + 'px';
        canvas.style.height = rowHeight + 'px';

        canvas.width = Math.floor(viewportWidth * dpr);
        canvas.height = Math.floor(rowHeight * dpr);

        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, viewportWidth, rowHeight);

        const tMin = Math.max(0, scrollLeft / scale);
        const tMax = (scrollLeft + viewportWidth) / scale;

        // Draw track vertical gridlines
        const tickInterval = getNiceTickInterval((viewportWidth / 10) / scale);
        const firstTick = Math.floor(tMin / tickInterval) * tickInterval;

        ctx.strokeStyle = '#f1f5f9';
        ctx.lineWidth = 1;

        for (let t = firstTick; t <= tMax; t += tickInterval) {
            const gx = t * scale - scrollLeft;
            ctx.beginPath();
            ctx.moveTo(gx, 0);
            ctx.lineTo(gx, rowHeight);
            ctx.stroke();
        }

        // Draw track items using binary search filtering
        const startIdx = findStartIndex(track.data, tMin);
        let lastPixelX = -1;

        if (track.type === 'engine') {
            for (let i = startIdx; i < track.data.length; i++) {
                const task = track.data[i];
                if (task.start > tMax) break;

                const isZero = task.duration === 0;
                const barW = isZero ? 24 : Math.max(task.duration * scale, 2);
                const drawX = isZero ? (task.start * scale - scrollLeft - 12) : (task.start * scale - scrollLeft);
                const drawY = 16;
                const drawH = 40;

                // Canvas pixel binning optimization
                if (!isZero && barW < 1.5) {
                    const currentPixelX = Math.floor(drawX);
                    if (currentPixelX === lastPixelX) continue;
                    lastPixelX = currentPixelX;
                }

                const color = getOpColor(task.op);
                const isHovered = (task === track.hoveredItem);

                if (isZero) {
                    ctx.fillStyle = '#f8fafc';
                    ctx.fillRect(drawX, drawY, barW, drawH);
                    ctx.save();
                    ctx.setLineDash([3, 3]);
                    ctx.strokeStyle = isHovered ? '#0f172a' : '#94a3b8';
                    ctx.lineWidth = isHovered ? 2 : 1;
                    ctx.strokeRect(drawX, drawY, barW, drawH);
                    ctx.restore();
                } else {
                    ctx.fillStyle = color.bg;
                    drawRoundedRect(ctx, drawX, drawY, barW, drawH, 4);
                    ctx.fill();

                    ctx.strokeStyle = isHovered ? '#0f172a' : color.border;
                    ctx.lineWidth = isHovered ? 2.5 : 1;
                    ctx.stroke();

                    if (barW > 25) {
                        ctx.fillStyle = color.text;
                        ctx.font = '700 11px Inter, -apple-system, sans-serif';
                        const text = getTruncatedText(ctx, task.name, barW - 10);
                        if (text) {
                            ctx.fillText(text, drawX + 6, drawY + 24);
                        }
                    }
                }
            }
        } else if (track.type === 'logical') {
            const numBufs = track.extra.numBufs;
            const pctHeight = 1 / numBufs;

            for (let i = startIdx; i < track.data.length; i++) {
                const buf = track.data[i];
                if (buf.start > tMax) break;

                const bIdx = i % numBufs;
                const drawX = buf.start * scale - scrollLeft;
                const drawW = Math.max((buf.end - buf.start) * scale, 2);

                const boxH = Math.max((rowHeight * pctHeight) - 4, 4);
                const boxY = rowHeight - ((bIdx + 1) * rowHeight * pctHeight) + 2;

                if (drawW < 1.5) {
                    const currentPixelX = Math.floor(drawX);
                    if (currentPixelX === lastPixelX) continue;
                    lastPixelX = currentPixelX;
                }

                const color = getOpColor(buf.op);
                const isHovered = (buf === track.hoveredItem);

                ctx.fillStyle = color.bg;
                drawRoundedRect(ctx, drawX, boxY, drawW, boxH, 3);
                ctx.fill();

                ctx.strokeStyle = isHovered ? '#0f172a' : color.border;
                ctx.lineWidth = isHovered ? 2.5 : 1;
                ctx.stroke();

                if (drawW > 25 && boxH > 14) {
                    ctx.fillStyle = color.text;
                    ctx.font = '700 10px Inter, -apple-system, sans-serif';
                    const text = getTruncatedText(ctx, buf.node_name, drawW - 8);
                    if (text) {
                        ctx.fillText(text, drawX + 4, boxY + boxH / 2 + 3);
                    }
                }
            }
        } else if (track.type === 'physical') {
            const maxOffsetAndSize = track.extra.maxOffsetAndSize;

            // Draw Y-axis guideline gridlines
            const numYGridlines = 4;
            const yStep = maxOffsetAndSize / numYGridlines;

            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = 'rgba(203, 213, 225, 0.7)';
            ctx.font = '600 9px Inter, -apple-system, sans-serif';
            ctx.fillStyle = '#94a3b8';

            for (let k = 0; k <= numYGridlines; k++) {
                const yVal = Math.round(k * yStep);
                const canvasY = rowHeight - (yVal / maxOffsetAndSize) * rowHeight;
                ctx.beginPath();
                ctx.moveTo(0, canvasY);
                ctx.lineTo(viewportWidth, canvasY);
                ctx.stroke();

                if (k < numYGridlines) {
                    ctx.fillText(`O: ${formatBytes(yVal)}`, 6, canvasY - 2);
                }
            }
            ctx.restore();

            for (let i = startIdx; i < track.data.length; i++) {
                const buf = track.data[i];
                if (buf.start > tMax) break;

                const bufOffset = buf.offset >= 0 ? buf.offset : 0;
                const pctHeight = buf.size / maxOffsetAndSize;
                const pctBottom = bufOffset / maxOffsetAndSize;

                const boxH = Math.max(pctHeight * rowHeight - 4, 4);
                const boxY = rowHeight - (pctBottom + pctHeight) * rowHeight + 2;
                const drawX = buf.start * scale - scrollLeft;
                const drawW = Math.max((buf.end - buf.start) * scale, 2);

                if (drawW < 1.5) {
                    const currentPixelX = Math.floor(drawX);
                    if (currentPixelX === lastPixelX) continue;
                    lastPixelX = currentPixelX;
                }

                const color = getOpColor(buf.op);
                const isHovered = (buf === track.hoveredItem);

                ctx.fillStyle = color.bg;
                drawRoundedRect(ctx, drawX, boxY, drawW, boxH, 3);
                ctx.fill();

                ctx.strokeStyle = isHovered ? '#0f172a' : color.border;
                ctx.lineWidth = isHovered ? 2.5 : 1;
                ctx.stroke();

                if (drawW > 25 && boxH > 14) {
                    ctx.fillStyle = color.text;
                    ctx.font = '700 10px Inter, -apple-system, sans-serif';
                    const text = getTruncatedText(ctx, buf.node_name, drawW - 8);
                    if (text) {
                        ctx.fillText(text, drawX + 4, boxY + boxH / 2 + 3);
                    }
                }
            }
        }

        ctx.restore();
    });
}

function renderRulerTicks(track, scrollLeft, viewportWidth) {
    const scale = track.scale;
    const tMin = Math.max(0, scrollLeft / scale);
    const tMax = (scrollLeft + viewportWidth) / scale;

    const tickInterval = getNiceTickInterval((viewportWidth / 10) / scale);
    const firstTick = Math.floor(tMin / tickInterval) * tickInterval;

    track.container.innerHTML = '';

    for (let t = firstTick; t <= tMax && t <= track.maxTime; t += tickInterval) {
        const tick = document.createElement('div');
        tick.className = 'ruler-tick';
        tick.style.left = (t * scale) + 'px';
        tick.textContent = Number.isInteger(t) ? t : t.toFixed(1);
        track.container.appendChild(tick);
    }
}

/**
 * Pointer event tracking for canvas item hover and Inspector updates.
 */
function setupHoverInteractivity(track) {
    const container = track.container;

    container.addEventListener('mousemove', (e) => {
        const rect = container.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const scale = track.scale;
        const hoverTime = mouseX / scale;

        let foundItem = null;

        if (track.type === 'engine') {
            const startIdx = findStartIndex(track.data, hoverTime - 10);
            for (let i = startIdx; i < track.data.length; i++) {
                const task = track.data[i];
                if (task.start > hoverTime + 10) break;

                const isZero = task.duration === 0;
                const startX = isZero ? (task.start * scale - 12) : (task.start * scale);
                const endX = isZero ? (task.start * scale + 12) : (task.end * scale);

                if (mouseX >= startX && mouseX <= endX && mouseY >= 16 && mouseY <= 56) {
                    foundItem = task;
                    break;
                }
            }
        } else if (track.type === 'logical') {
            const numBufs = track.extra.numBufs;
            const pctHeight = 1 / numBufs;
            const rowHeight = track.rowHeight;

            const startIdx = findStartIndex(track.data, hoverTime - 10);
            for (let i = startIdx; i < track.data.length; i++) {
                const buf = track.data[i];
                if (buf.start > hoverTime + 10) break;

                const bIdx = i % numBufs;
                const startX = buf.start * scale;
                const endX = buf.end * scale;
                const boxH = Math.max((rowHeight * pctHeight) - 4, 4);
                const boxY = rowHeight - ((bIdx + 1) * rowHeight * pctHeight) + 2;

                if (mouseX >= startX && mouseX <= endX && mouseY >= boxY && mouseY <= boxY + boxH) {
                    foundItem = buf;
                    break;
                }
            }
        } else if (track.type === 'physical') {
            const maxOffsetAndSize = track.extra.maxOffsetAndSize;
            const rowHeight = track.rowHeight;

            const startIdx = findStartIndex(track.data, hoverTime - 10);
            for (let i = startIdx; i < track.data.length; i++) {
                const buf = track.data[i];
                if (buf.start > hoverTime + 10) break;

                const bufOffset = buf.offset >= 0 ? buf.offset : 0;
                const pctHeight = buf.size / maxOffsetAndSize;
                const pctBottom = bufOffset / maxOffsetAndSize;

                const boxH = Math.max(pctHeight * rowHeight - 4, 4);
                const boxY = rowHeight - (pctBottom + pctHeight) * rowHeight + 2;
                const startX = buf.start * scale;
                const endX = buf.end * scale;

                if (mouseX >= startX && mouseX <= endX && mouseY >= boxY && mouseY <= boxY + boxH) {
                    foundItem = buf;
                    break;
                }
            }
        }

        if (foundItem !== track.hoveredItem) {
            track.hoveredItem = foundItem;
            container.style.cursor = foundItem ? 'pointer' : 'default';

            if (foundItem) {
                if (track.type === 'engine') {
                    showInspector(foundItem, track.extra.formatted);
                } else {
                    showMemInspector(foundItem, track.extra.spaceName, track.extra.handleName);
                }
            }

            requestViewportRender();
        }
    });

    container.addEventListener('mouseleave', () => {
        if (track.hoveredItem) {
            track.hoveredItem = null;
            container.style.cursor = 'default';
            requestViewportRender();
        }
    });
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

if (graphSelect && graphSelect.value) {
    loadGraphData(graphSelect.value);
}