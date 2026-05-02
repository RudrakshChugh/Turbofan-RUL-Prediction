import React, { useState, useEffect, useRef } from 'react';

// === NavBar Component ===
function NavBar() {
  const [time, setTime] = useState(new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
  
  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'space-between', 
      alignItems: 'center', 
      padding: '16px 28px', 
      borderBottom: '1px solid var(--ink)',
      backgroundColor: 'var(--paper)'
    }}>
      <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '12px' }}>
        RR/AEGIS · Predictive Maintenance
      </div>
      <div style={{ display: 'flex', gap: '32px' }}>
        <a href="#" style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '3px', color: 'var(--ink)', textDecoration: 'none' }}>Dashboard</a>
        <a href="#" style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '3px', color: 'var(--ink)', textDecoration: 'none' }}>Fleet View</a>
        <a href="#" style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '3px', color: 'var(--ink)', textDecoration: 'none' }}>System</a>
      </div>
      <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span>{time}</span>
        <span style={{ color: 'var(--ink3)' }}>|</span>
        <span>v2.4.1</span>
        <span style={{ color: 'var(--green)' }}>● LIVE</span>
      </div>
    </div>
  );
}

// === FleetStrip Component ===
const mockFleet = [
  { id: 'FD001-01', rul: 112, status: 'green' },
  { id: 'FD001-02', rul: 84, status: 'green' },
  { id: 'FD001-03', rul: 38, status: 'amber' },
  { id: 'FD001-04', rul: 12, status: 'red' },
  { id: 'FD001-05', rul: 95, status: 'green' },
  { id: 'FD001-06', rul: 125, status: 'green' },
  { id: 'FD001-07', rul: 41, status: 'amber' },
  { id: 'FD001-08', rul: 76, status: 'green' },
];

function FleetStrip() {
  const [selected, setSelected] = useState('FD001-03');

  return (
    <div style={{ 
      display: 'flex', 
      overflowX: 'auto', 
      backgroundColor: 'var(--paper)',
      paddingBottom: '20px'
    }}>
      {mockFleet.map(engine => {
        const isSelected = selected === engine.id;
        const colorVar = `var(--${engine.status})`;
        
        return (
          <div 
            key={engine.id}
            onClick={() => setSelected(engine.id)}
            style={{
              padding: '16px 24px',
              borderRight: '1px solid var(--rule)',
              borderTop: isSelected ? '2px solid var(--ink)' : `2px solid ${colorVar}`,
              backgroundColor: isSelected ? 'var(--panel)' : 'transparent',
              minWidth: '140px',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px'
            }}
          >
            <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '11px', fontWeight: 500 }}>
              {engine.id}
            </div>
            <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '22px', fontWeight: 500, color: colorVar }}>
              {engine.rul}
            </div>
            <div>
              <div style={{ height: '3px', backgroundColor: 'var(--rule)', width: '100%', marginBottom: '6px' }}>
                <div style={{ height: '100%', width: `${Math.min(100, (engine.rul / 125) * 100)}%`, backgroundColor: 'var(--ink)' }}></div>
              </div>
              <div className="label" style={{ color: colorVar }}>
                {engine.status === 'red' ? 'CRITICAL' : engine.status === 'amber' ? 'WATCH' : 'NOMINAL'}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// === DegradationChart Component ===
function DegradationChart() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    let animationId;
    
    // Setup for High-DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    
    const w = rect.width;
    const h = rect.height;
    const padX = 0; 
    const padY = 20;
    const plotW = w - padX * 2;
    const plotH = h - padY * 2 - 40;
    
    const maxCycles = 250;
    const currentCycle = 138;
    const failureThreshold = 0.3;
    
    const histData = [];
    for(let i=0; i<=currentCycle; i++) {
       const val = 1.0 - (Math.pow(i/maxCycles, 2)) * 0.4 + (Math.sin(i*0.1) * 0.01);
       histData.push({x: i, y: val});
    }
    
    const predData = [];
    const ciUpper = [];
    const ciLower = [];
    for(let i=currentCycle; i<=maxCycles; i++) {
       const stepsAhead = i - currentCycle;
       const val = histData[currentCycle].y - Math.pow(stepsAhead/100, 1.5) * 0.5;
       predData.push({x: i, y: val});
       ciUpper.push({x: i, y: val + stepsAhead * 0.002});
       ciLower.push({x: i, y: val - stepsAhead * 0.003});
    }
    
    const mapX = (val) => padX + (val / maxCycles) * plotW;
    const mapY = (val) => padY + plotH - ((val - 0.1) / 0.95) * plotH;
    
    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      
      // 1. CI Band
      ctx.fillStyle = 'rgba(184,112,32,0.12)';
      ctx.beginPath();
      ctx.moveTo(mapX(predData[0].x), mapY(predData[0].y));
      for(let i=1; i<ciUpper.length; i++) ctx.lineTo(mapX(ciUpper[i].x), mapY(ciUpper[i].y));
      for(let i=ciLower.length-1; i>=0; i--) ctx.lineTo(mapX(ciLower[i].x), mapY(ciLower[i].y));
      ctx.closePath();
      ctx.fill();
      
      // 2. Failure threshold
      ctx.strokeStyle = '#C0302A';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padX, mapY(failureThreshold));
      ctx.lineTo(padX + plotW, mapY(failureThreshold));
      ctx.stroke();
      ctx.setLineDash([]);
      
      // 3. Historical line
      ctx.strokeStyle = '#0D0D0E';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(mapX(histData[0].x), mapY(histData[0].y));
      for(let i=1; i<histData.length; i++) ctx.lineTo(mapX(histData[i].x), mapY(histData[i].y));
      ctx.stroke();
      
      // 4. Predicted continuation
      ctx.strokeStyle = '#0D0D0E';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(mapX(predData[0].x), mapY(predData[0].y));
      for(let i=1; i<predData.length; i++) ctx.lineTo(mapX(predData[i].x), mapY(predData[i].y));
      ctx.stroke();
      ctx.setLineDash([]);
      
      // 5. "now" marker
      const nowX = mapX(currentCycle);
      ctx.strokeStyle = '#8A8A8D';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(nowX, padY - 10);
      ctx.lineTo(nowX, padY + plotH + 10);
      ctx.stroke();
      
      ctx.fillStyle = '#8A8A8D';
      ctx.font = '8px "IBM Plex Mono"';
      ctx.textAlign = 'center';
      ctx.fillText('now', nowX, padY - 14);
      
      // 6. Failure intersection
      let intersectCycle = maxCycles;
      for(let i=0; i<predData.length; i++) {
         if (predData[i].y <= failureThreshold) {
            intersectCycle = predData[i].x;
            break;
         }
      }
      const intX = mapX(intersectCycle);
      const intY = mapY(failureThreshold);
      ctx.strokeStyle = '#C0302A';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(intX, intY, 4, 0, Math.PI * 2);
      ctx.stroke();
      
      // X-axis bar
      const barY = padY + plotH + 30;
      ctx.fillStyle = '#DCDCDE';
      ctx.fillRect(padX, barY, plotW, 3);
      ctx.fillStyle = '#0D0D0E';
      ctx.fillRect(padX, barY, (currentCycle / maxCycles) * plotW, 3);
    };

    draw();

    return () => cancelAnimationFrame(animationId);
  }, []);

  return (
    <div style={{ width: '100%', height: '100%', minHeight: '400px', display: 'flex', flexDirection: 'column' }}>
      <canvas 
        ref={canvasRef} 
        style={{ width: '100%', height: '100%', flex: 1, display: 'block' }}
      />
    </div>
  );
}

// === SensorTable Component ===
const mockSensors = [
  { id: 'T24', name: 'LPC outlet temp', val: 642.35, unit: '°R', norm: 0.8, drift: 3.2 },
  { id: 'T30', name: 'HPC outlet temp', val: 1589.7, unit: '°R', norm: 0.6, drift: 1.5 },
  { id: 'T50', name: 'LPT outlet temp', val: 1412.1, unit: '°R', norm: 0.9, drift: 4.1 },
  { id: 'P30', name: 'HPC outlet press', val: 553.12, unit: 'psia', norm: 0.4, drift: 0.2 },
  { id: 'Nf', name: 'Physical fan speed', val: 2388.1, unit: 'rpm', norm: 0.5, drift: 0.8 },
  { id: 'Nc', name: 'Physical core speed', val: 9052.4, unit: 'rpm', norm: 0.7, drift: 1.1 },
  { id: 'Ps30', name: 'HPC outlet stat press', val: 47.45, unit: 'psia', norm: 0.3, drift: 0.1 },
];

function SensorTable() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {mockSensors.map((s, i) => {
        let statusColor = 'var(--ink3)';
        let dotColor = 'var(--rule)';
        if (s.drift > 3) {
          statusColor = 'var(--red)';
          dotColor = 'var(--red)';
        } else if (s.drift > 1) {
          statusColor = 'var(--amber)';
          dotColor = 'var(--amber)';
        }

        return (
          <div 
            key={i}
            style={{ 
              display: 'grid', 
              gridTemplateColumns: '1.4fr 1fr 1fr 20px', 
              alignItems: 'center',
              padding: '10px 0 10px 12px',
              borderBottom: '1px solid var(--rule)',
              transition: 'all 0.15s ease',
              borderLeft: '0.5px solid transparent'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'var(--paper)';
              e.currentTarget.style.borderLeft = '0.5px solid var(--ink)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderLeft = '0.5px solid transparent';
            }}
          >
            <div>
              <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '10px' }}>{s.id}</div>
              <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '10px', color: 'var(--ink3)' }}>{s.name}</div>
            </div>
            <div style={{ paddingRight: '16px' }}>
              <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '10px', fontWeight: 500, marginBottom: '6px' }}>
                {s.val} <span style={{ color: 'var(--ink3)', fontWeight: 400 }}>{s.unit}</span>
              </div>
              <div style={{ height: '3px', backgroundColor: 'var(--rule)', width: '100%' }}>
                <div style={{ height: '100%', width: `${s.norm * 100}%`, backgroundColor: 'var(--ink)' }}></div>
              </div>
            </div>
            <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '10px', color: statusColor, textAlign: 'right', paddingRight: '12px' }}>
              +{s.drift}%
            </div>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: dotColor }}></div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// === UncertaintyPanel Component ===
function UncertaintyPanel() {
  const data = [
    { label: 'Aleatoric', value: 8.2, rangeStart: 2, rangeEnd: 15 },
    { label: 'Epistemic', value: 4.1, rangeStart: 1, rangeEnd: 8 },
    { label: 'Combined', value: 12.3, rangeStart: 3, rangeEnd: 23 },
  ];
  
  const maxScale = 40;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {data.map((row, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ width: '70px', fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px', color: 'var(--ink2)' }}>
            {row.label}
          </div>
          <div style={{ flex: 1, height: '16px', backgroundColor: 'var(--paper)', position: 'relative' }}>
            {/* CI Band */}
            <div style={{ 
              position: 'absolute', 
              top: 0, 
              bottom: 0, 
              left: `${(row.rangeStart / maxScale) * 100}%`, 
              width: `${((row.rangeEnd - row.rangeStart) / maxScale) * 100}%`,
              backgroundColor: 'rgba(180,150,100,0.18)'
            }}></div>
            {/* Estimate Point */}
            <div style={{
              position: 'absolute',
              top: '5px',
              left: `calc(${(row.value / maxScale) * 100}% - 3px)`,
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: 'var(--ink)'
            }}></div>
          </div>
          <div style={{ width: '30px', textAlign: 'right', fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px' }}>
            {row.value.toFixed(1)}
          </div>
        </div>
      ))}
    </div>
  );
}

// === DomainPanel Component ===
function DomainPanel() {
  const domains = ['FD001', 'FD002', 'FD003', 'FD004'];
  const active = 'FD002';
  const source = 'FD001';
  const mmd = 0.14;

  return (
    <div>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
        {domains.map(d => {
          const isActive = d === active;
          const isSource = d === source;
          return (
            <div key={d} style={{
              padding: '6px 10px',
              border: isActive ? '0.5px solid var(--ink)' : '0.5px solid var(--rule)',
              color: isActive ? 'var(--ink)' : 'var(--ink3)',
              fontFamily: '"IBM Plex Mono", monospace',
              fontSize: '9px',
              letterSpacing: '1px'
            }}>
              {d}{isSource ? ' · Source' : ''}
            </div>
          );
        })}
      </div>
      <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px', color: 'var(--ink3)' }}>
        Adapted via MMD · shift Δ = <span style={{ color: mmd > 0.1 ? 'var(--amber)' : 'var(--ink)' }}>{mmd}</span>
      </div>
    </div>
  );
}

// === App Component ===
function App() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <NavBar />
      <div style={{ backgroundColor: 'var(--paper)', flex: 1, display: 'flex', flexDirection: 'column' }}>
        <FleetStrip />
        <div className="multi-pane" style={{ gridTemplateColumns: '1fr 400px', flex: 1 }}>
           <div className="pane" style={{ padding: '20px 28px 20px 0', display: 'flex', flexDirection: 'column' }}>
              <h2 className="section-header" style={{ marginLeft: '28px' }}>Degradation Trajectory</h2>
              <div style={{ flex: 1, marginLeft: '28px' }}>
                 <DegradationChart />
              </div>
           </div>
           <div className="pane" style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ padding: '0 28px 28px 20px' }}>
                 <h2 className="section-header">Sensor Telemetry</h2>
                 <SensorTable />
              </div>
              <div style={{ padding: '0 28px 28px 20px' }}>
                 <h2 className="section-header">Uncertainty Analysis</h2>
                 <UncertaintyPanel />
              </div>
              <div style={{ padding: '0 28px 28px 20px' }}>
                 <h2 className="section-header">Domain Adaptation</h2>
                 <DomainPanel />
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}

export default App;
