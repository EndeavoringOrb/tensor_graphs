/**
 * Lightweight EGraph Viewer
 * Handles eclass selection, graph extraction, and SVG rendering
 */

class EGraphViewer {
    constructor() {
        this.currentFile = null;
        this.egraphData = null;
        this.selectionMap = {}; // { eclass_id: enode_id }
        this.graphData = { nodes: [], edges: [] };
        this.searchTerm = ''; // New: Store search term

        // SVG state
        this.svg = null;
        this.zoomLevel = 1;
        this.pan = { x: 0, y: 0 };
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };

        // Layout constants
        this.NODE_RADIUS = { eclass: 20, enode: 18 };
        this.SPACING_X = 180;
        this.SPACING_Y = 80;

        this.init();
    }

    init() {
        this.svg = document.getElementById('graph-svg');
        this.setupEventListeners();
        this.renderArrowMarker();
    }

    setupEventListeners() {
        // File selection
        document.getElementById('file-select').addEventListener('change', (e) => {
            this.loadFile(e.target.value);
        });

        // Search input
        document.getElementById('search-input').addEventListener('input', (e) => {
            this.searchTerm = e.target.value.toLowerCase();
            this.renderEclassList();
        });

        // Controls
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSelections();
        });
        document.getElementById('zoom-in-btn').addEventListener('click', () => {
            this.zoom(1.2);
        });
        document.getElementById('zoom-out-btn').addEventListener('click', () => {
            this.zoom(0.8);
        });

        // SVG pan/zoom
        this.svg.addEventListener('wheel', (e) => this.handleWheel(e));
        this.svg.addEventListener('mousedown', (e) => this.handlePanStart(e));
        document.addEventListener('mousemove', (e) => this.handlePanMove(e));
        document.addEventListener('mouseup', () => this.handlePanEnd());
    }

    async loadFile(filename) {
        if (!filename) return;

        this.currentFile = filename;
        this.selectionMap = {};
        this.searchTerm = ''; // Reset search on new file
        document.getElementById('search-input').value = '';

        try {
            const response = await fetch(`/api/egraph/${filename}`);
            this.egraphData = await response.json();
            this.renderEclassList();
            this.updateSelectionDisplay();
            this.extractAndRenderGraph();
        } catch (err) {
            console.error('Failed to load egraph:', err);
            document.getElementById('eclass-list').innerHTML =
                '<p class="placeholder">Error loading file</p>';
        }
    }

    renderEclassList() {
        const container = document.getElementById('eclass-list');
        const { eclasses } = this.egraphData;

        if (!eclasses || Object.keys(eclasses).length === 0) {
            container.innerHTML = '<p class="placeholder">No eclasses found</p>';
            return;
        }

        // Sort eclasses by ID
        const sortedIds = Object.keys(eclasses).map(Number).sort((a, b) => a - b);

        container.innerHTML = sortedIds.map(eclassId => {
            const eclass = eclasses[eclassId];
            const enodes = eclass.enodes || [];

            // Filter enodes based on search term
            const filteredEnodes = enodes.filter(enodeId => {
                const enode = this.egraphData.enodes[enodeId];
                const opName = enode?.op_name || 'Unknown';
                return opName.toLowerCase().includes(this.searchTerm);
            });

            // If search term is present and no enodes match, hide the eclass
            if (this.searchTerm && filteredEnodes.length === 0) {
                return '';
            }

            const isSelected = this.selectionMap[eclassId] !== undefined;

            return `
        <div class="eclass-item" data-eclass-id="${eclassId}">
          <div class="eclass-header" onclick="viewer.toggleEclass(${eclassId})">
            <span class="eclass-id">EC${eclassId}</span>
            <span class="enode-count">${filteredEnodes.length} enode${filteredEnodes.length !== 1 ? 's' : ''}</span>
          </div>
          <div class="enodes-list ${isSelected ? 'expanded' : ''}" id="enodes-${eclassId}">
            ${filteredEnodes.map(enodeId => {
                const enode = this.egraphData.enodes[enodeId];
                const isEnodeSelected = this.selectionMap[eclassId] === enodeId;
                return `
                <div class="enode-item ${isEnodeSelected ? 'selected' : ''}" 
                     data-enode-id="${enodeId}"
                     data-eclass-id="${eclassId}"
                     onclick="viewer.selectEnode(${eclassId}, ${enodeId})">
                  <span class="dot ${isEnodeSelected ? 'selected' : 'enode'}"></span>
                  <span class="enode-id">EN${enodeId}</span>
                  <span class="op-name">${enode?.op_name || 'Unknown'}</span>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
        }).join('');
    }

    toggleEclass(eclassId) {
        const list = document.getElementById(`enodes-${eclassId}`);
        if (list) {
            list.classList.toggle('expanded');
        }
    }

    selectEnode(eclassId, enodeId) {
        // Toggle selection
        if (this.selectionMap[eclassId] === enodeId) {
            delete this.selectionMap[eclassId];
        } else {
            this.selectionMap[eclassId] = enodeId;
        }

        // Re-render eclass list to update selection styling
        this.renderEclassList();
        this.updateSelectionDisplay();

        // Extract and render updated graph
        this.extractAndRenderGraph();
    }

    resetSelections() {
        this.selectionMap = {};
        this.renderEclassList();
        this.updateSelectionDisplay();
        this.extractAndRenderGraph();
    }

    updateSelectionDisplay() {
        const display = document.getElementById('selection-map-display');
        const entries = Object.entries(this.selectionMap);

        if (entries.length === 0) {
            display.textContent = 'None';
        } else {
            display.textContent = entries
                .map(([ec, en]) => `EC${ec}→EN${en}`)
                .join(', ');
        }
    }

    async extractAndRenderGraph() {
        if (!this.currentFile) {
            document.getElementById('graph-empty').style.display = 'block';
            this.clearGraph();
            return;
        }

        try {
            const response = await fetch('/api/extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: this.currentFile,
                    selection_map: this.selectionMap
                })
            });

            this.graphData = await response.json();
            this.renderGraph();
        } catch (err) {
            console.error('Failed to extract graph:', err);
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

        // Compute layout using hierarchical approach
        const layout = this.computeLayout(nodes, edges);

        // Create SVG group for transforms
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('id', 'graph-group');
        g.setAttribute('transform', `translate(${this.pan.x}, ${this.pan.y}) scale(${this.zoomLevel})`);
        this.svg.appendChild(g);

        // Draw edges first (so they appear behind nodes)
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
                this.drawNode(g, node, pos, this.selectionMap[node.eclass_id] === node.enode_id);
            }
        });

        // Set SVG viewBox
        this.svg.setAttribute('viewBox', `${layout.bounds.x - 50} ${layout.bounds.y - 50} ${layout.bounds.width + 100} ${layout.bounds.height + 100}`);
    }

    computeLayout(nodes, edges) {
        // Simple hierarchical layout: group by depth, position horizontally
        const byDepth = {};
        nodes.forEach(node => {
            const depth = node.depth || 0;
            if (!byDepth[depth]) byDepth[depth] = [];
            byDepth[depth].push(node);
        });

        const positions = {};
        let maxY = 0;

        // Position nodes by depth level
        Object.keys(byDepth).sort((a, b) => a - b).forEach(depthStr => {
            const depth = parseInt(depthStr);
            const levelNodes = byDepth[depth];
            const levelWidth = levelNodes.length * this.SPACING_X;
            const startX = -levelWidth / 2 + this.SPACING_X / 2;

            levelNodes.forEach((node, i) => {
                positions[node.id] = {
                    x: startX + i * this.SPACING_X,
                    y: depth * this.SPACING_Y
                };
            });

            maxY = Math.max(maxY, depth * this.SPACING_Y);
        });

        // Compute bounds
        const xs = Object.values(positions).map(p => p.x);
        const ys = Object.values(positions).map(p => p.y);
        const minX = Math.min(...xs), maxX = Math.max(...xs);
        const minY = Math.min(...ys), maxY_val = Math.max(...ys);

        return {
            nodes: positions,
            bounds: {
                x: minX,
                y: minY,
                width: maxX - minX,
                height: maxY_val - minY
            }
        };
    }

    drawNode(group, node, pos, isSelected) {
        const isEclass = node.type === 'eclass';
        const radius = isEclass ? this.NODE_RADIUS.eclass : this.NODE_RADIUS.enode;
        const className = `node ${node.type}${isSelected ? ' selected' : ''}`;

        // Circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pos.x);
        circle.setAttribute('cy', pos.y);
        circle.setAttribute('r', radius);
        circle.setAttribute('class', className);
        circle.setAttribute('data-node-id', node.id);
        circle.addEventListener('click', (e) => {
            e.stopPropagation();
            this.handleNodeClick(node);
        });
        group.appendChild(circle);

        // Label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', pos.x);
        label.setAttribute('y', pos.y + (isEclass ? 0 : 2));
        label.setAttribute('class', 'node-label');
        label.textContent = node.label;
        group.appendChild(label);

        // Tooltip (title)
        circle.addEventListener('mouseenter', (e) => this.showTooltip(e, node));
        circle.addEventListener('mouseleave', () => this.hideTooltip());
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

    handleNodeClick(node) {
        // If clicking an eclass, expand it in sidebar
        if (node.type === 'eclass') {
            this.toggleEclass(node.eclass_id);
        }
    }

    // Tooltip handling
    showTooltip(event, node) {
        // Simple title attribute for now; could be enhanced with custom tooltip
        event.target.setAttribute('title',
            `${node.label}\nType: ${node.type}\nID: ${node.eclass_id || node.enode_id}`
        );
    }

    hideTooltip() {
        // No-op for native title tooltips
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
        if (event.button !== 0) return; // Only left mouse button
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

// Initialize viewer when DOM is ready
let viewer;
document.addEventListener('DOMContentLoaded', () => {
    viewer = new EGraphViewer();
});