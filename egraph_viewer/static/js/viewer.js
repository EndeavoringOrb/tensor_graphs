/* static/js/viewer.js */
/**
 * Incremental EGraph Explorer
 * Click eclass → see enodes in sidebar → select enode → expand children
 */

class EGraphViewer {
    constructor() {
        this.currentFile = null;
        this.egraphMeta = null;
        this.selectionMap = {};      // {eclass_id: enode_id}
        this.childCache = {};
        this.visibleEclasses = new Set();
        this.focusedEclass = null;
        this.focusedEclassData = null;
        this.focusedEnodes = [];
        this.navigationPath = [];    // For breadcrumb
        this.graphData = { nodes: [], edges: [] };
        this.settings = { constant_limit: 3 };

        // SVG state
        this.svg = null;
        this.zoomLevel = 1;
        this.pan = { x: 0, y: 0 };
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };

        // Layout
        this.NODE_RADIUS = { eclass: 24, enode: 20 };
        this.SPACING_X = 160;
        this.SPACING_Y = 80;

        this.init();
    }

    init() {
        this.svg = document.getElementById('graph-svg');
        this.setupEventListeners();
        this.renderArrowMarker();
        this.loadSettings();
    }

    setupEventListeners() {
        document.getElementById('file-select').addEventListener('change', (e) => {
            this.loadFile(e.target.value);
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSelections();
        });
        document.getElementById('zoom-in-btn').addEventListener('click', () => {
            this.zoom(1.2);
        });
        document.getElementById('zoom-out-btn').addEventListener('click', () => {
            this.zoom(0.8);
        });

        // Settings modal
        document.getElementById('settings-btn').addEventListener('click', () => {
            this.openSettingsModal();
        });
        document.getElementById('settings-close').addEventListener('click', () => {
            this.closeSettingsModal();
        });
        document.getElementById('settings-cancel').addEventListener('click', () => {
            this.closeSettingsModal();
        });
        document.getElementById('settings-save').addEventListener('click', () => {
            this.saveSettings();
        });
        document.getElementById('settings-modal').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) {
                this.closeSettingsModal();
            }
        });

        this.svg.addEventListener('wheel', (e) => this.handleWheel(e));
        this.svg.addEventListener('mousedown', (e) => this.handlePanStart(e));
        document.addEventListener('mousemove', (e) => this.handlePanMove(e));
        document.addEventListener('mouseup', () => this.handlePanEnd());
    }

    async loadSettings() {
        try {
            const response = await fetch('/api/settings');
            this.settings = await response.json();
        } catch (err) {
            console.error('Failed to load settings:', err);
        }
    }

    openSettingsModal() {
        document.getElementById('constant-limit-input').value = this.settings.constant_limit;
        document.getElementById('settings-modal').style.display = 'flex';
    }

    closeSettingsModal() {
        document.getElementById('settings-modal').style.display = 'none';
    }

    async saveSettings() {
        const limit = parseInt(document.getElementById('constant-limit-input').value, 10);
        if (isNaN(limit) || limit < 1) {
            document.getElementById('settings-error').textContent = 'Must be a positive integer';
            return;
        }
        if (limit > 1000) {
            document.getElementById('settings-error').textContent = 'Maximum 1000 elements';
            return;
        }

        document.getElementById('settings-error').textContent = '';

        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ constant_limit: limit })
            });
            this.settings = await response.json();
            this.closeSettingsModal();

            // Refresh focused eclass to get updated constants
            if (this.focusedEclass !== null) {
                await this.focusEclass(this.focusedEclass);
            }
        } catch (err) {
            console.error('Failed to save settings:', err);
            document.getElementById('settings-error').textContent = 'Failed to save settings';
        }
    }

    updateVisibleEclasses() {
        const newVisible = new Set();

        // 1. Always show the root
        if (this.egraphMeta) {
            newVisible.add(this.egraphMeta.root_eclass);
        }

        // 2. Always show the currently focused e-class
        if (this.focusedEclass !== null) {
            newVisible.add(this.focusedEclass);
        }

        // 3. Show everything in the selection map and their children
        for (const ecId in this.selectionMap) {
            newVisible.add(parseInt(ecId));

            // Add children associated with this selection from our cache
            const children = this.childCache[ecId] || [];
            children.forEach(childId => newVisible.add(childId));
        }

        this.visibleEclasses = newVisible;
    }

    async loadFile(filename) {
        if (!filename) return;

        this.currentFile = filename;
        this.selectionMap = {};
        this.visibleEclasses = new Set();
        this.focusedEclass = null;
        this.focusedEclassData = null;
        this.focusedEnodes = [];
        this.navigationPath = [];
        this.childCache = {};

        try {
            const response = await fetch(`/api/egraph/${filename}`);
            this.egraphMeta = await response.json();

            document.getElementById('file-info').textContent =
                `${this.egraphMeta.num_eclasses.toLocaleString()} eclasses, ${this.egraphMeta.num_enodes.toLocaleString()} enodes`;

            this.visibleEclasses.add(this.egraphMeta.root_eclass);
            await this.focusEclass(this.egraphMeta.root_eclass);
        } catch (err) {
            console.error('Failed to load egraph:', err);
            document.getElementById('eclass-list').innerHTML =
                '<p class="placeholder">Error loading file</p>';
        }
    }

    async focusEclass(eclassId) {
        if (this.focusedEclass === eclassId) return;

        this.focusedEclass = eclassId;

        // Update navigation path
        const existingIdx = this.navigationPath.indexOf(eclassId);
        if (existingIdx >= 0) {
            this.navigationPath = this.navigationPath.slice(0, existingIdx + 1);
        } else {
            this.navigationPath.push(eclassId);
        }

        try {
            const response = await fetch(`/api/eclass/${this.currentFile}/${eclassId}`);
            if (!response.ok) {
                this.focusedEnodes = [];
                this.focusedEclassData = null;
                this.updateSidebar();
                return;
            }
            const data = await response.json();
            console.log(data);
            this.focusedEclassData = data;
            this.focusedEnodes = data.enodes || [];
            this.updateVisibleEclasses();
            this.updateSidebar();
            this.updateBreadcrumb();
            this.fetchAndRenderGraph();
        } catch (err) {
            console.error('Failed to load eclass:', err);
            this.focusedEnodes = [];
            this.focusedEclassData = null;
            this.updateSidebar();
        }
    }

    async selectEnode(eclassId, enodeId, children) {
        if (this.selectionMap[eclassId] === enodeId) {
            // Deselect
            delete this.selectionMap[eclassId];
            delete this.childCache[eclassId];
        } else {
            this.selectionMap[eclassId] = enodeId;
            this.childCache[eclassId] = children || [];
        }

        this.updateVisibleEclasses();
        this.updateSidebar();
        this.fetchAndRenderGraph();
    }

    resetSelections() {
        this.selectionMap = {};
        this.childCache = {};
        this.focusedEclass = this.egraphMeta.root_eclass;
        this.navigationPath = [this.egraphMeta.root_eclass];

        this.updateVisibleEclasses();
        this.updateSidebar();
        this.updateBreadcrumb();
        this.fetchAndRenderGraph();
    }

    updateBreadcrumb() {
        const container = document.getElementById('breadcrumb');
        if (!this.currentFile || this.navigationPath.length === 0) {
            container.innerHTML = '';
            return;
        }

        container.innerHTML = this.navigationPath.map((eclassId, idx) => {
            const isLast = idx === this.navigationPath.length - 1;
            const sep = idx > 0 ? '<span class="breadcrumb-sep">›</span>' : '';
            const cls = isLast ? 'breadcrumb-item active' : 'breadcrumb-item';
            return `${sep}<span class="${cls}" onclick="viewer.focusEclass(${eclassId})">EC${eclassId}</span>`;
        }).join('');
    }

    formatConstantValue(value, dtype) {
        if (value === null || value === undefined) return '—';

        if (typeof value === 'boolean') {
            return value ? 'true' : 'false';
        }

        if (typeof value === 'number') {
            if (dtype === 'FLOAT32' || dtype === 'BF16') {
                return value.toFixed(6);
            }
            return value.toString();
        }

        if (typeof value === 'string') {
            return value; // Already formatted (e.g., hex for BF16)
        }

        return String(value);
    }

    renderConstantSection(constant, dtype) {
        if (!constant) return '';

        const { values, total_count, is_truncated } = constant;

        let html = `
            <div class="constant-section">
                <div class="constant-header">
                    <span class="constant-title">📦 Constant Data</span>
                    <span class="constant-meta">${total_count.toLocaleString()} elements</span>
                </div>
                <div class="constant-values">
        `;

        if (values.length === 0) {
            html += '<span class="constant-empty">No data</span>';
        } else {
            values.forEach((val, idx) => {
                const formatted = this.formatConstantValue(val, dtype);
                html += `<span class="constant-value">${formatted}</span>`;
                if (idx < values.length - 1) {
                    html += '<span class="constant-sep">,</span>';
                }
            });

            if (is_truncated) {
                html += `<span class="constant-more">, ... (+${(total_count - values.length).toLocaleString()} more)</span>`;
            }
        }

        html += `
                </div>
                ${is_truncated ? `<div class="constant-truncated-note">Showing first ${values.length} of ${total_count.toLocaleString()} elements</div>` : ''}
            </div>
        `;

        return html;
    }

    updateSidebar() {
        const container = document.getElementById('eclass-list');

        if (!this.currentFile) {
            container.innerHTML = '<p class="placeholder">Select a file to start exploring</p>';
            return;
        }

        if (this.focusedEclass === null) {
            container.innerHTML = `
                <p class="placeholder">
                    Click on an eclass node in the graph to see its enodes<br><br>
                    <strong>💡 Tip:</strong> Start with EC${this.egraphMeta.root_eclass} (the root)
                </p>`;
            return;
        }

        const data = this.focusedEclassData;
        const selectedEnodeId = this.selectionMap[this.focusedEclass];
        const isExpanded = selectedEnodeId !== undefined;

        let html = `
            <div class="focused-eclass-header">
                <span class="eclass-id">EClass ${this.focusedEclass}</span>
                <span class="enode-count">${this.focusedEnodes.length} enode${this.focusedEnodes.length !== 1 ? 's' : ''}</span>
            </div>
            <div class="eclass-meta">
                <div class="meta-row">
                    <span class="meta-label">Shape:</span>
                    <span class="meta-value">[${data.shape.join(', ')}]</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">DType:</span>
                    <span class="meta-value dtype-badge">${data.dtype}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Backend:</span>
                    <span class="meta-value backend-badge">${data.backend}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Strides:</span>
                    <span class="meta-value">[${data.strides.join(', ')}]</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Offset:</span>
                    <span class="meta-value">${data.view_offset}</span>
                </div>
            </div>
        `;

        // Add constant section if present
        if (data.constant) {
            html += this.renderConstantSection(data.constant, data.dtype);
        }

        html += `<div class="enodes-list">`;
        html += `<div class="enodes-list-header">ENodes</div>`;

        if (this.focusedEnodes.length === 0) {
            html += '<p class="placeholder">No enodes found</p>';
        } else {
            this.focusedEnodes.forEach(enode => {
                const isSelected = selectedEnodeId === enode.id;
                const childrenStr = JSON.stringify(enode.children);
                html += `
                    <div class="enode-item ${isSelected ? 'selected' : ''}"
                         onclick="viewer.selectEnode(${this.focusedEclass}, ${enode.id}, ${childrenStr})">
                        <span class="dot ${isSelected ? 'selected' : 'enode'}"></span>
                        <span class="enode-id">EN${enode.id}</span>
                        <span class="op-name">${enode.op_name}</span>
                        <span class="child-count">${enode.children.length}↓</span>
                    </div>
                `;
            });
        }

        html += '</div>';
        container.innerHTML = html;
    }

    updateSelectionDisplay() {
        const display = document.getElementById('selection-map-display');
        const countEl = document.getElementById('visible-count');
        const entries = Object.entries(this.selectionMap);

        if (entries.length === 0) {
            display.textContent = 'None';
        } else {
            const displayEntries = entries.slice(0, 5).map(([ec, en]) => `EC${ec}→EN${en}`);
            if (entries.length > 5) {
                displayEntries.push(`...+${entries.length - 5} more`);
            }
            display.textContent = displayEntries.join(', ');
        }

        if (countEl) {
            countEl.textContent = `${this.visibleEclasses.size} eclasses visible`;
        }
    }

    async fetchAndRenderGraph() {
        if (!this.currentFile) {
            this.clearGraph();
            document.getElementById('graph-empty').style.display = 'block';
            return;
        }

        this.updateSelectionDisplay();

        try {
            const response = await fetch('/api/explore', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: this.currentFile,
                    selection_map: this.selectionMap,
                    visible_eclasses: Array.from(this.visibleEclasses)
                })
            });

            this.graphData = await response.json();
            this.renderGraph();
        } catch (err) {
            console.error('Failed to fetch graph:', err);
        }
    }

    clearGraph() {
        while (this.svg.firstChild) {
            this.svg.removeChild(this.svg.firstChild);
        }
        this.renderArrowMarker();
        this.zoomLevel = 1;
        this.pan = { x: 0, y: 0 };
    }

    renderGraph() {
        const { nodes, edges } = this.graphData;

        if (nodes.length === 0) {
            document.getElementById('graph-empty').style.display = 'block';
            this.clearGraph();
            return;
        }

        document.getElementById('graph-empty').style.display = 'none';
        this.clearGraph();

        const layout = this.computeLayout(nodes, edges);

        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('id', 'graph-group');
        g.setAttribute('transform', `translate(${this.pan.x}, ${this.pan.y}) scale(${this.zoomLevel})`);
        this.svg.appendChild(g);

        // Draw edges first
        edges.forEach(edge => {
            const source = layout.nodes[edge.source];
            const target = layout.nodes[edge.target];
            if (source && target) {
                this.drawEdge(g, source, target, edge.type);
            }
        });

        // Draw nodes
        nodes.forEach(node => {
            const pos = layout.nodes[node.id];
            if (pos) {
                const isExpanded = node.type === 'eclass' && this.selectionMap[node.eclass_id] !== undefined;
                const isSelectedEnode = node.type === 'enode' && this.selectionMap[node.eclass_id] === node.enode_id;
                this.drawNode(g, node, pos, isExpanded, isSelectedEnode);
            }
        });

        // Center view on graph
        const bounds = layout.bounds;
        this.svg.setAttribute('viewBox',
            `${bounds.x - 60} ${bounds.y - 60} ${bounds.width + 120} ${bounds.height + 120}`);
    }

    computeLayout(nodes, edges) {
        // Build depth using BFS from root
        const depths = {};
        const childTargets = new Set(edges.filter(e => e.type === 'child').map(e => e.target));
        const rootCandidates = nodes.filter(n => !childTargets.has(n.id) && n.type === 'eclass');
        const root = rootCandidates[0];

        if (!root) {
            nodes.forEach(n => depths[n.id] = 0);
        } else {
            const queue = [{ id: root.id, depth: 0 }];
            depths[root.id] = 0;

            while (queue.length > 0) {
                const { id, depth } = queue.shift();
                edges.filter(e => e.source === id).forEach(edge => {
                    if (!(edge.target in depths)) {
                        depths[edge.target] = depth + 1;
                        queue.push({ id: edge.target, depth: depth + 1 });
                    }
                });
            }
        }

        // Group by depth and track parent for better horizontal positioning
        const byDepth = {};
        nodes.forEach(node => {
            const depth = depths[node.id] || 0;
            if (!byDepth[depth]) byDepth[depth] = [];
            byDepth[depth].push(node);
        });

        const positions = {};

        // Assign positions using subtree-aware spacing
        Object.keys(byDepth).sort((a, b) => a - b).forEach(depthStr => {
            const levelNodes = byDepth[depthStr];
            const levelWidth = levelNodes.length * this.SPACING_X;
            const startX = -levelWidth / 2 + this.SPACING_X / 2;

            levelNodes.forEach((node, i) => {
                positions[node.id] = {
                    x: startX + i * this.SPACING_X,
                    y: (depths[node.id] || 0) * this.SPACING_Y
                };
            });
        });

        const xs = Object.values(positions).map(p => p.x);
        const ys = Object.values(positions).map(p => p.y);
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY = Math.max(...ys);

        return {
            nodes: positions,
            bounds: {
                x: minX,
                y: minY,
                width: Math.max(maxX - minX, 100),
                height: Math.max(maxY - minY, 100)
            }
        };
    }

    drawNode(group, node, pos, isExpanded, isSelectedEnode) {
        const isEclass = node.type === 'eclass';
        const radius = isEclass ? this.NODE_RADIUS.eclass : this.NODE_RADIUS.enode;

        let className = `node ${node.type}`;
        if (isExpanded) className += ' expanded';
        if (isSelectedEnode) className += ' selected';

        // Circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pos.x);
        circle.setAttribute('cy', pos.y);
        circle.setAttribute('r', radius);
        circle.setAttribute('class', className);
        circle.setAttribute('data-node-id', node.id);

        circle.addEventListener('click', (e) => {
            e.stopPropagation();
            if (isEclass) {
                this.focusEclass(node.eclass_id);
            } else {
                // Click on enode focuses its parent eclass
                this.focusEclass(node.eclass_id);
            }
        });

        group.appendChild(circle);

        // Label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', pos.x);
        label.setAttribute('y', pos.y + (isEclass ? 1 : 2));
        label.setAttribute('class', `node-label ${isEclass ? 'eclass-label' : ''}`);

        // Truncate long labels
        let labelText = node.label;
        if (labelText.length > 10 && !isEclass) {
            labelText = labelText.substring(0, 9) + '…';
        }
        label.textContent = labelText;
        group.appendChild(label);

        // Expanded indicator for eclasses
        if (isExpanded && isEclass) {
            const indicator = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            indicator.setAttribute('x', pos.x + radius + 4);
            indicator.setAttribute('y', pos.y - radius + 4);
            indicator.setAttribute('font-size', '10px');
            indicator.setAttribute('fill', '#28a745');
            indicator.textContent = '✓';
            indicator.style.pointerEvents = 'none';
            group.appendChild(indicator);
        }
    }

    drawEdge(group, source, target, type) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', source.x);
        line.setAttribute('y1', source.y);
        line.setAttribute('x2', target.x);
        line.setAttribute('y2', target.y);
        line.setAttribute('class', `edge ${type}`);
        group.appendChild(line);
    }

    renderArrowMarker() {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrowhead');
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '7');
        marker.setAttribute('refX', '10');
        marker.setAttribute('refY', '3.5');
        marker.setAttribute('orient', 'auto');

        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
        polygon.setAttribute('fill', '#6c757d');

        marker.appendChild(polygon);
        defs.appendChild(marker);
        this.svg.appendChild(defs);
    }

    // Zoom handling
    zoom(factor) {
        this.zoomLevel = Math.max(0.2, Math.min(3, this.zoomLevel * factor));
        this.updateTransform();
    }

    handleWheel(event) {
        event.preventDefault();
        const factor = event.deltaY > 0 ? 0.9 : 1.1;
        this.zoom(factor);
    }

    // Pan handling
    handlePanStart(event) {
        if (event.button !== 0) return;
        this.isDragging = true;
        this.dragStart = { x: event.clientX - this.pan.x, y: event.clientY - this.pan.y };
        this.svg.style.cursor = 'grabbing';
    }

    handlePanMove(event) {
        if (!this.isDragging) return;
        this.pan.x = event.clientX - this.dragStart.x;
        this.pan.y = event.clientY - this.dragStart.y;
        this.updateTransform();
    }

    handlePanEnd() {
        this.isDragging = false;
        this.svg.style.cursor = 'default';
    }

    updateTransform() {
        const group = document.getElementById('graph-group');
        if (group) {
            group.setAttribute('transform', `translate(${this.pan.x}, ${this.pan.y}) scale(${this.zoomLevel})`);
        }
    }
}

let viewer;
document.addEventListener('DOMContentLoaded', () => {
    viewer = new EGraphViewer();
});