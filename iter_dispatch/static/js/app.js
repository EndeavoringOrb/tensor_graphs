let currentGraphOrders = [];
let orderAIndex = 0;
let orderBIndex = 0;

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

/**
 * Handles fetching graph payload and setting up application states.
 */
async function loadGraphData(name) {
    if (!name) return;
    try {
        const resp = await fetch(`/api/graph/${encodeURIComponent(name)}`);
        const data = await resp.json();

        // Map the serialized JSON strings from the new API format into JS objects
        currentGraphOrders = data.orders.map(orderObj => {
            const parsedNodes = orderObj.ordered.map(str => JSON.parse(str));
            const parsedBuffers = orderObj.buffers.map(str => JSON.parse(str));
            const parsedAllocated = orderObj.allocated.map(str => JSON.parse(str));

            // Reconstruct buffer-to-node association to resolve node names for allocations
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
                const engineKey = node.engine ? `Engine(idx=${node.engine.idx}, engine_type=${node.engine.engine_type})` : 'Engine(idx=0, engine_type=EngineType.CPU)';
                return {
                    name: node.name,
                    op: opName,
                    start: node.birth,
                    end: node.birth + node.cost,
                    duration: node.cost,
                    engine: engineKey,
                    size: node.size
                };
            });

            // Map parsed buffers to the raw buffer structure
            const buffers = parsedBuffers.map(buf => {
                const node = bufferIdxToNode[buf.idx];
                const nodeName = node ? node.name : `Buf ${buf.idx}`;
                const opName = node ? node.op.replace('Op.', '') : 'INPUT';
                const memSpaceIdx = buf.mem_space ? buf.mem_space.idx : 0;
                const memSpaceHandle = buf.mem_space && buf.mem_space.handle_type ? buf.mem_space.handle_type.replace('Handle.', '') : '';

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

            // Map parsed allocated buffers to the allocated block structure
            const allocated = parsedAllocated.map(buf => {
                const node = bufferIdxToNode[buf.idx];
                const nodeName = node ? node.name : `Buf ${buf.idx}`;
                const opName = node ? node.op.replace('Op.', '') : 'INPUT';
                const memSpaceIdx = buf.mem_space ? buf.mem_space.idx : 0;
                const memSpaceHandle = buf.mem_space && buf.mem_space.handle_type ? buf.mem_space.handle_type.replace('Handle.', '') : '';

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

        // Show control panels
        graphStatsPanel.classList.remove('hidden');
        nav.classList.remove('hidden');

        // Calculate statistics and set default selections
        calculateStats();

        // Render both schedules
        renderComparison();
    } catch (e) {
        console.error("Error fetching execution schedules: ", e);
    }
}

// Automatically load on dropdown modifications
graphSelect.addEventListener('change', (e) => {
    loadGraphData(e.target.value);
});

// Select order dropdown event listeners
selectOrderA.addEventListener('change', (e) => {
    orderAIndex = parseInt(e.target.value);
    updateCostBadges();
    renderComparison();
});

// Select order dropdown event listeners
selectOrderB.addEventListener('change', (e) => {
    orderBIndex = parseInt(e.target.value);
    updateCostBadges();
    renderComparison();
});

// Step navigation buttons
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

/**
 * Strips verbose Python class names down to a cleaner format.
 */
function formatEngineName(rawName) {
    const idxMatch = rawName.match(/idx=(\d+)/);
    const typeMatch = rawName.match(/EngineType\.(\w+)/);
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

/**
 * Calculates graph-wide statistics (highest/lowest costs)
 * and defaults order selection indexes to that pair.
 */
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
    statBestOrder.textContent = `Order ${minIdx + 1} (Cost: ${minCost})`;
    statWorstOrder.textContent = `Order ${maxIdx + 1} (Cost: ${maxCost})`;

    // Default to lowest and highest cost orders
    orderAIndex = minIdx;
    orderBIndex = maxIdx;

    populateDropdowns(minIdx, maxIdx);
}

/**
 * Populates dropdown lists with available orders
 */
function populateDropdowns(defaultA, defaultB) {
    selectOrderA.innerHTML = '';
    selectOrderB.innerHTML = '';

    currentGraphOrders.forEach((orderObj, idx) => {
        const schedule = orderObj.schedule;
        const cost = schedule.length > 0 ? Math.max(...schedule.map(t => t.end)) : 0;

        const optionA = document.createElement('option');
        optionA.value = idx;
        optionA.textContent = `Order ${idx + 1} (Cost: ${cost})`;
        selectOrderA.appendChild(optionA);

        const optionB = document.createElement('option');
        optionB.value = idx;
        optionB.textContent = `Order ${idx + 1} (Cost: ${cost})`;
        selectOrderB.appendChild(optionB);
    });

    selectOrderA.value = defaultA;
    selectOrderB.value = defaultB;

    updateCostBadges();
}

/**
 * Updates quick cost stats displayed on selector boxes
 */
function updateCostBadges() {
    const costAElement = document.getElementById('totalCostA');
    const costBElement = document.getElementById('totalCostB');

    const orderAObj = currentGraphOrders[orderAIndex];
    const orderBObj = currentGraphOrders[orderBIndex];

    const scheduleA = orderAObj ? orderAObj.schedule : [];
    const scheduleB = orderBObj ? orderBObj.schedule : [];

    const costA = scheduleA && scheduleA.length > 0 ? Math.max(...scheduleA.map(t => t.end)) : 0;
    const costB = scheduleB && scheduleB.length > 0 ? Math.max(...scheduleB.map(t => t.end)) : 0;

    costAElement.textContent = costA;
    costBElement.textContent = costB;
}

/**
 * Syncs the pager button states based on current selection index boundaries.
 */
function updateButtonStates() {
    prevBtnA.disabled = (orderAIndex === 0);
    nextBtnA.disabled = (orderAIndex === currentGraphOrders.length - 1);
    prevBtnB.disabled = (orderBIndex === 0);
    nextBtnB.disabled = (orderBIndex === currentGraphOrders.length - 1);
}

/**
 * Renders both chosen schedules aligned horizontally inside the timeline panel.
 */
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

    // Determine the common timeline scale max value across both schedules
    const maxTimeA = scheduleA.length > 0 ? Math.max(...scheduleA.map(t => t.end)) : 0;
    const maxTimeB = scheduleB.length > 0 ? Math.max(...scheduleB.map(t => t.end)) : 0;
    const commonMaxTime = Math.max(maxTimeA, maxTimeB, 1);

    // Pixel scaling factor
    const scale = 50;

    // Render Schedule A Block
    const blockA = document.createElement('div');
    blockA.className = 'schedule-block schedule-block-a';
    renderScheduleInto(blockA, scheduleA, buffersA, allocatedA, orderAIndex, commonMaxTime, scale, 'A');
    vizContainer.appendChild(blockA);

    // Separator line
    const divider = document.createElement('div');
    divider.className = 'schedule-divider';
    vizContainer.appendChild(divider);

    // Render Schedule B Block
    const blockB = document.createElement('div');
    blockB.className = 'schedule-block schedule-block-b';
    renderScheduleInto(blockB, scheduleB, buffersB, allocatedB, orderBIndex, commonMaxTime, scale, 'B');
    vizContainer.appendChild(blockB);
}

/**
 * Injects ruler, grids, timeline lanes, and malloc memory blocks into a container.
 */
function renderScheduleInto(container, order, buffers, allocated, orderIndex, maxTime, scale, blockLabel) {
    // Calculate the physical layout width needed for the timeline
    const timelineWidthStr = (maxTime * scale) + 'px';

    // Block Title Row
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
    costValue.textContent = cost;
    headerRow.appendChild(costValue);

    container.appendChild(headerRow);

    // Block Timings Canvas
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

    // 1. Build Time Ruler row
    const rulerRow = document.createElement('div');
    rulerRow.className = 'ruler-row';

    const rulerSpacer = document.createElement('div');
    rulerSpacer.className = 'ruler-label-spacer';
    rulerRow.appendChild(rulerSpacer);

    const rulerTicks = document.createElement('div');
    rulerTicks.className = 'ruler-ticks';
    rulerTicks.style.width = timelineWidthStr;
    rulerTicks.style.flexShrink = '0';

    // Scale boundaries depending on time sizes
    const tickInterval = maxTime > 30 ? 5 : (maxTime > 15 ? 2 : 1);

    for (let t = 0; t <= maxTime; t += tickInterval) {
        const tick = document.createElement('div');
        tick.className = 'ruler-tick';
        tick.style.left = (t * scale) + 'px';
        tick.textContent = t;
        rulerTicks.appendChild(tick);
    }
    rulerRow.appendChild(rulerTicks);
    blockBody.appendChild(rulerRow);

    // 2. Build aligned backing gridline overlay
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

    // 3. Populate physical hardware track rows
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

            bar.className = `task-bar op-${task.op}`;
            if (isZero) {
                bar.classList.add('task-zero-duration');
            }

            bar.style.left = offsetLeft + 'px';
            bar.style.width = barWidth + 'px';
            bar.textContent = task.name;

            bar.title = `${task.name} (${task.op}): Time ${task.start}-${task.end} (duration: ${task.duration})`;

            // Connect task properties on hover directly to detailsPanel Inspector
            bar.addEventListener('mouseenter', () => {
                showInspector(task, formatted);
            });
            bar.addEventListener('mouseleave', () => {
                hideInspector();
            });

            timeline.appendChild(bar);
        });

        row.appendChild(label);
        row.appendChild(timeline);
        blockBody.appendChild(row);
    });

    // 4. Populate Logical Memory Tracks (lifetimes)
    if (buffers && buffers.length > 0) {
        // Section Header for logical lifetimes
        const memSectionHeader = document.createElement('div');
        memSectionHeader.className = 'mem-section-header';
        memSectionHeader.innerHTML = '<span>Logical Buffer Lifetimes (Active Spans)</span>';
        blockBody.appendChild(memSectionHeader);

        // Group by memory space index
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
                bar.className = `mem-buffer-bar op-${buf.op}`;

                const barWidth = (buf.end - buf.start) * scale;
                const offsetLeft = buf.start * scale;

                bar.style.left = offsetLeft + 'px';
                bar.style.width = barWidth + 'px';

                // Stacking them vertically based on count
                const pctHeight = 100 / numBufs;
                const pctBottom = bIdx * pctHeight;

                bar.style.height = `calc(${pctHeight}% - 4px)`;
                bar.style.bottom = `calc(${pctBottom}% + 2px)`;

                bar.innerHTML = `<span class="mem-buffer-label">${buf.node_name}</span>`;
                bar.title = `Buffer ${buf.idx} (${buf.node_name}): Active ${buf.start}-${buf.end}, Size: ${buf.size}`;

                bar.addEventListener('mouseenter', () => {
                    showMemInspector(buf, spaceName, handleName);
                });
                bar.addEventListener('mouseleave', () => {
                    hideInspector();
                });

                memTimeline.appendChild(bar);
            });

            row.appendChild(label);
            row.appendChild(memTimeline);
            blockBody.appendChild(row);
        });
    }

    // 5. Populate Physical Memory Tracks (allocated offsets)
    // Section Header for physical allocation layout
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

            // Calculate dynamic Y scaling boundary (max offset + size)
            const maxOffsetAndSize = Math.max(...spaceBuffers.map(b => b.offset + b.size), 1);

            // Draw Y-axis guideline gridlines and numeric step ticks
            for (let y = 0; y <= maxOffsetAndSize; y++) {
                const line = document.createElement('div');
                line.className = 'mem-offset-gridline';
                line.style.bottom = `${(y / maxOffsetAndSize) * 100}%`;
                memTimeline.appendChild(line);

                if (y < maxOffsetAndSize) {
                    const tickText = document.createElement('span');
                    tickText.className = 'mem-offset-tick-text';
                    tickText.textContent = `O: ${y}`;
                    tickText.style.bottom = `${(y / maxOffsetAndSize) * 100 + 1}%`;
                    memTimeline.appendChild(tickText);
                }
            }

            // Draw buffers inside 2D space
            spaceBuffers.forEach(buf => {
                const bar = document.createElement('div');
                bar.className = `mem-buffer-bar op-${buf.op}`;

                const barWidth = (buf.end - buf.start) * scale;
                const offsetLeft = buf.start * scale;

                bar.style.left = offsetLeft + 'px';
                bar.style.width = barWidth + 'px';

                // Percentage height & vertical positioning
                const pctHeight = (buf.size / maxOffsetAndSize) * 100;
                const pctBottom = (buf.offset / maxOffsetAndSize) * 100;

                bar.style.height = `calc(${pctHeight}% - 4px)`;
                bar.style.bottom = `calc(${pctBottom}% + 2px)`;

                bar.innerHTML = `<span class="mem-buffer-label">${buf.node_name}</span>`;
                bar.title = `Buffer ${buf.idx} (${buf.node_name}): Active ${buf.start}-${buf.end}, Offset: ${buf.offset}, Size: ${buf.size}`;

                bar.addEventListener('mouseenter', () => {
                    showMemInspector(buf, spaceName, handleName);
                });
                bar.addEventListener('mouseleave', () => {
                    hideInspector();
                });

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

    inspectOp.className = 'inspect-badge';
    inspectOp.classList.add(`op-${task.op}`);
    inspectOp.textContent = task.op;

    inspectName.textContent = `Node: ${task.name}`;
    inspectEngine.textContent = `${engineDetails.name} (${engineDetails.sub})`;
    inspectInterval.textContent = `${task.start} → ${task.end}`;
    inspectDuration.textContent = `${task.duration} unit${task.duration !== 1 ? 's' : ''}`;
}

function showMemInspector(buf, spaceName, handleName) {
    detailsPanel.classList.remove('hidden');

    inspectOp.className = 'inspect-badge';
    inspectOp.classList.add(`op-${buf.op}`);
    inspectOp.textContent = buf.op;

    inspectName.textContent = `Buffer: ${buf.node_name}`;
    inspectEngine.textContent = `${spaceName} (${handleName})`;
    inspectInterval.textContent = `Active: ${buf.start} → ${buf.end}`;
    inspectDuration.textContent = `Offset: ${buf.offset} | Size: ${buf.size}`;
}

function hideInspector() {
    // Keep last hovered item details displayed to avoid jittery page reflows
}

// Initial execution load
if (graphSelect && graphSelect.value) {
    loadGraphData(graphSelect.value);
}