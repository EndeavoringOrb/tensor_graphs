let currentGraphOrders = [];
let currentIndex = 0;

const graphSelect = document.getElementById('graphSelect');
const loadBtn = document.getElementById('loadBtn');
const vizContainer = document.getElementById('vizContainer');
const orderInfo = document.getElementById('orderInfo');
const totalCostSpan = document.getElementById('totalCost');
const nav = document.getElementById('navigation');

loadBtn.addEventListener('click', async () => {
    const name = graphSelect.value;
    const resp = await fetch(`/api/graph/${name}`);
    const data = await resp.json();
    
    currentGraphOrders = data.orders;
    currentIndex = 0;
    nav.classList.remove('hidden');
    renderOrder();
});

document.getElementById('nextBtn').addEventListener('click', () => {
    if (currentIndex < currentGraphOrders.length - 1) {
        currentIndex++;
        renderOrder();
    }
});

document.getElementById('prevBtn').addEventListener('click', () => {
    if (currentIndex > 0) {
        currentIndex--;
        renderOrder();
    }
});

function renderOrder() {
    const order = currentGraphOrders[currentIndex];
    const totalOrders = currentGraphOrders.length;
    orderInfo.textContent = `Order ${currentIndex + 1} of ${totalOrders}`;
    
    // Clear previous
    vizContainer.innerHTML = '';
    
    // Group tasks by engine
    const engines = [...new Set(order.map(t => t.engine))].sort();
    const maxTime = Math.max(...order.map(t => t.end));
    totalCostSpan.textContent = maxTime;

    // Scale factor (px per unit of cost)
    const scale = 40; 

    engines.forEach(engName => {
        const row = document.createElement('div');
        row.className = 'engine-row';
        
        const label = document.createElement('div');
        label.className = 'engine-label';
        label.textContent = engName;
        
        const timeline = document.createElement('div');
        timeline.className = 'timeline';
        
        const engineTasks = order.filter(t => t.engine === engName);
        engineTasks.forEach(task => {
            const bar = document.createElement('div');
            bar.className = `task-bar op-${task.op}`;
            bar.style.left = (task.start * scale) + 'px';
            bar.style.width = Math.max((task.duration * scale), 2) + 'px';
            bar.title = `${task.name} (${task.op}): ${task.start}-${task.end}`;
            bar.textContent = task.name;
            timeline.appendChild(bar);
        });

        row.appendChild(label);
        row.appendChild(timeline);
        vizContainer.appendChild(row);
    });
}