import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8000';

// === NavBar Component ===
function NavBar({ currentView, setView }) {
  const [time, setTime] = useState(new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
  
  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const navLinkStyle = (viewName) => ({
    fontFamily: '"IBM Plex Mono", monospace',
    fontSize: '9px',
    textTransform: 'uppercase',
    letterSpacing: '3px',
    color: currentView === viewName ? 'var(--ink)' : 'var(--ink3)',
    textDecoration: currentView === viewName ? 'underline' : 'none',
    textUnderlineOffset: '4px',
    cursor: 'pointer',
    background: 'none',
    border: 'none',
    padding: 0
  });

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
        <button style={navLinkStyle('dashboard')} onClick={() => setView('dashboard')}>Dashboard</button>
        <button style={navLinkStyle('fleet')} onClick={() => setView('fleet')}>Fleet View</button>
        <button style={navLinkStyle('system')} onClick={() => setView('system')}>System</button>
      </div>
      <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span>{time}</span>
        <span style={{ color: 'var(--ink3)' }}>|</span>
        <span>v2.4.1</span>
        <span style={{ color: 'var(--ink3)' }}>● STATIC</span>
      </div>
    </div>
  );
}

// === FleetStrip Component ===
function FleetStrip({ fleet, selectedIndex, onSelect }) {
  const stripRef = useRef(null);

  const scroll = (direction) => {
    if (stripRef.current) {
      stripRef.current.scrollBy({ left: direction * 300, behavior: 'smooth' });
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'stretch', backgroundColor: 'var(--paper)' }}>
      {/* Left arrow */}
      <button 
        onClick={() => scroll(-1)}
        style={{
          border: 'none',
          background: 'var(--paper)',
          cursor: 'pointer',
          padding: '0 8px',
          fontFamily: '"IBM Plex Mono", monospace',
          fontSize: '16px',
          color: 'var(--ink3)',
          borderRight: '1px solid var(--rule)',
          flexShrink: 0,
        }}
        aria-label="Scroll fleet left"
      >◂</button>

      <div 
        ref={stripRef}
        style={{ 
          display: 'flex', 
          overflowX: 'auto',
          flex: 1,
          scrollbarWidth: 'none',
        }}
      >
        {fleet.map((engine) => {
          const isSelected = selectedIndex === engine.index;
          const colorVar = `var(--${engine.status})`;
          
          return (
            <div 
              key={engine.id}
              onClick={() => onSelect(engine.index)}
              style={{
                padding: '16px 24px',
                borderRight: '1px solid var(--rule)',
                borderTop: isSelected ? '2px solid var(--ink)' : `2px solid ${colorVar}`,
                backgroundColor: isSelected ? 'var(--panel)' : 'transparent',
                minWidth: '140px',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px',
                transition: 'background-color 0.15s ease',
              }}
            >
              <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '11px', fontWeight: 500 }}>
                {engine.id}
              </div>
              <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '22px', fontWeight: 500, color: colorVar }}>
                {engine.rul.toFixed(1)}
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

      {/* Right arrow */}
      <button 
        onClick={() => scroll(1)}
        style={{
          border: 'none',
          background: 'var(--paper)',
          cursor: 'pointer',
          padding: '0 8px',
          fontFamily: '"IBM Plex Mono", monospace',
          fontSize: '16px',
          color: 'var(--ink3)',
          borderLeft: '1px solid var(--rule)',
          flexShrink: 0,
        }}
        aria-label="Scroll fleet right"
      >▸</button>
    </div>
  );
}

// === DegradationChart Component ===
function DegradationChart({ engineDetail }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !engineDetail) return;
    
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
    
    const traj = engineDetail.trajectory;
    const pred = engineDetail.prediction;
    
    const totalCycles = engineDetail.totalCycles;
    const maxCycles = pred.cycles.length > 0 ? pred.cycles[pred.cycles.length - 1] : totalCycles + 50;
    const failureThreshold = traj.failureThreshold;
    
    // Build data points
    const histData = traj.cycles.map((c, i) => ({ x: c, y: traj.healthIndex[i] }));
    const predData = pred.cycles.map((c, i) => ({ x: c, y: pred.healthIndex[i] }));
    const ciUpper = pred.cycles.map((c, i) => ({ x: c, y: pred.ciUpper[i] }));
    const ciLower = pred.cycles.map((c, i) => ({ x: c, y: pred.ciLower[i] }));
    
    // Y range
    const allY = [
      ...histData.map(d => d.y),
      ...predData.map(d => d.y),
      ...ciUpper.map(d => d.y),
      ...ciLower.map(d => d.y),
      failureThreshold,
    ];
    const yMin = Math.min(...allY) - 0.05;
    const yMax = Math.max(...allY) + 0.05;
    
    const mapX = (val) => padX + (val / maxCycles) * plotW;
    const mapY = (val) => padY + plotH - ((val - yMin) / (yMax - yMin)) * plotH;
    
    ctx.clearRect(0, 0, w, h);
    
    // 1. CI Band
    if (predData.length > 1) {
      ctx.fillStyle = 'rgba(184,112,32,0.12)';
      ctx.beginPath();
      ctx.moveTo(mapX(predData[0].x), mapY(predData[0].y));
      for (let i = 1; i < ciUpper.length; i++) ctx.lineTo(mapX(ciUpper[i].x), mapY(ciUpper[i].y));
      for (let i = ciLower.length - 1; i >= 0; i--) ctx.lineTo(mapX(ciLower[i].x), mapY(ciLower[i].y));
      ctx.closePath();
      ctx.fill();
    }
    
    // 2. Failure threshold
    ctx.strokeStyle = '#C0302A';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padX, mapY(failureThreshold));
    ctx.lineTo(padX + plotW, mapY(failureThreshold));
    ctx.stroke();
    ctx.setLineDash([]);

    // Threshold label
    ctx.fillStyle = '#C0302A';
    ctx.font = '8px "IBM Plex Mono"';
    ctx.textAlign = 'left';
    ctx.fillText('failure threshold', padX + 4, mapY(failureThreshold) - 4);
    
    // 3. Historical line (smoothed health index)
    if (histData.length > 1) {
      ctx.strokeStyle = '#0D0D0E';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(mapX(histData[0].x), mapY(histData[0].y));
      for (let i = 1; i < histData.length; i++) ctx.lineTo(mapX(histData[i].x), mapY(histData[i].y));
      ctx.stroke();
    }
    
    // 4. Predicted continuation (dashed)
    if (predData.length > 1) {
      ctx.strokeStyle = '#0D0D0E';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(mapX(predData[0].x), mapY(predData[0].y));
      for (let i = 1; i < predData.length; i++) ctx.lineTo(mapX(predData[i].x), mapY(predData[i].y));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // 5. "now" marker
    const nowX = mapX(totalCycles);
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
    for (let i = 0; i < predData.length; i++) {
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
    ctx.fillRect(padX, barY, (totalCycles / maxCycles) * plotW, 3);

    // RUL annotation
    ctx.fillStyle = '#3A3A3C';
    ctx.font = '9px "IBM Plex Mono"';
    ctx.textAlign = 'left';
    ctx.fillText(`RUL ~${Math.round(engineDetail.rul)} cycles  ±${Math.round(engineDetail.uncertainty)}`, nowX + 8, padY + 4);

  }, [engineDetail]);

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
function SensorTable({ sensors }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      {sensors.map((s, i) => {
        let statusColor = 'var(--ink3)';
        let dotColor = 'var(--rule)';
        const absDrift = Math.abs(s.drift);
        if (absDrift > 5) {
          statusColor = 'var(--red)';
          dotColor = 'var(--red)';
        } else if (absDrift > 2) {
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
              {s.drift >= 0 ? '+' : ''}{s.drift}%
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
function UncertaintyPanel({ uncertaintyData }) {
  const data = [
    { label: uncertaintyData.epistemic.label, ...uncertaintyData.epistemic },
    { label: uncertaintyData.predictive.label, ...uncertaintyData.predictive },
    { label: uncertaintyData.conservative.label, ...uncertaintyData.conservative },
  ];
  
  const maxScale = Math.max(40, ...data.map(d => d.rangeEnd * 1.2));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {data.map((row, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ width: '90px', fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px', color: 'var(--ink2)' }}>
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
          <div style={{ width: '35px', textAlign: 'right', fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px' }}>
            {row.value.toFixed(1)}
          </div>
        </div>
      ))}
    </div>
  );
}

// === DomainPanel Component ===
function DomainPanel({ domainData }) {
  const domains = Array.from({ length: domainData.totalDomains }, (_, i) => `OC-${i}`);
  const activeIndex = domainData.id;

  return (
    <div>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
        {domains.map((d, i) => {
          const isActive = i === activeIndex;
          return (
            <div key={d} style={{
              padding: '6px 10px',
              border: isActive ? '0.5px solid var(--ink)' : '0.5px solid var(--rule)',
              color: isActive ? 'var(--ink)' : 'var(--ink3)',
              fontFamily: '"IBM Plex Mono", monospace',
              fontSize: '9px',
              letterSpacing: '1px'
            }}>
              {d}{isActive ? ' · Active' : ''}
            </div>
          );
        })}
      </div>
      <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '9px', color: 'var(--ink3)' }}>
        Adapted via DANN · Operating Condition Cluster {activeIndex} / {domainData.totalDomains}
      </div>
    </div>
  );
}


// === Views ===

function DashboardView({ fleet, selectedIndex, setSelectedIndex, engineDetail }) {
  return (
    <div style={{ backgroundColor: 'var(--paper)', flex: 1, display: 'flex', flexDirection: 'column' }}>
      <FleetStrip fleet={fleet} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
      <div className="multi-pane" style={{ gridTemplateColumns: '1fr 400px', flex: 1 }}>
         <div className="pane" style={{ padding: '20px 28px 20px 0', display: 'flex', flexDirection: 'column' }}>
            <h2 className="section-header" style={{ marginLeft: '28px' }}>Degradation Trajectory</h2>
            <div style={{ flex: 1, marginLeft: '28px' }}>
               <DegradationChart engineDetail={engineDetail} />
            </div>
         </div>
         <div className="pane" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '0 28px 28px 20px' }}>
               <h2 className="section-header">Sensor Telemetry</h2>
               {engineDetail ? <SensorTable sensors={engineDetail.sensors} /> : null}
            </div>
            <div style={{ padding: '0 28px 28px 20px' }}>
               <h2 className="section-header">Uncertainty Analysis</h2>
               {engineDetail ? <UncertaintyPanel uncertaintyData={engineDetail.uncertaintyDecomposition} /> : null}
            </div>
            <div style={{ padding: '0 28px 28px 20px' }}>
               <h2 className="section-header">Domain Adaptation</h2>
               {engineDetail ? <DomainPanel domainData={engineDetail.domain} /> : null}
            </div>
         </div>
      </div>
    </div>
  );
}

function FleetView({ fleet }) {
  const [sortConfig, setSortConfig] = useState({ key: 'rul', direction: 'ascending' });

  const sortedFleet = React.useMemo(() => {
    let sortableItems = [...fleet];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'ascending' ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'ascending' ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableItems;
  }, [fleet, sortConfig]);

  const requestSort = (key) => {
    let direction = 'ascending';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  const getSortIndicator = (columnName) => {
    if (sortConfig?.key === columnName) {
       return sortConfig.direction === 'ascending' ? ' ▲' : ' ▼';
    }
    return '';
  }

  const thStyle = {
    fontFamily: '"IBM Plex Mono", monospace',
    fontSize: '11px',
    fontWeight: 500,
    color: 'var(--ink3)',
    textTransform: 'uppercase',
    letterSpacing: '1px',
    padding: '16px 24px',
    textAlign: 'left',
    borderBottom: '1px solid var(--rule)',
    cursor: 'pointer',
    userSelect: 'none',
  };

  const tdStyle = {
    fontFamily: '"IBM Plex Mono", monospace',
    fontSize: '13px',
    padding: '16px 24px',
    borderBottom: '1px solid var(--rule)',
  };

  return (
    <div style={{ backgroundColor: 'var(--paper)', flex: 1, display: 'flex', flexDirection: 'column', padding: '32px' }}>
      <h2 className="section-header" style={{ marginBottom: '24px' }}>Fleet Overview ({fleet.length} units)</h2>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle} onClick={() => requestSort('id')}>Engine ID{getSortIndicator('id')}</th>
              <th style={thStyle} onClick={() => requestSort('status')}>Status{getSortIndicator('status')}</th>
              <th style={thStyle} onClick={() => requestSort('rul')}>Predicted RUL{getSortIndicator('rul')}</th>
              <th style={thStyle} onClick={() => requestSort('conservativeRul')}>Conservative RUL{getSortIndicator('conservativeRul')}</th>
              <th style={thStyle} onClick={() => requestSort('uncertainty')}>Uncertainty (±){getSortIndicator('uncertainty')}</th>
              <th style={thStyle} onClick={() => requestSort('trueRul')}>True RUL{getSortIndicator('trueRul')}</th>
            </tr>
          </thead>
          <tbody>
            {sortedFleet.map((engine) => {
              const colorVar = `var(--${engine.status})`;
              return (
                <tr key={engine.id} style={{ transition: 'background-color 0.15s ease' }} 
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--panel)'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}>
                  <td style={{ ...tdStyle, fontWeight: 500 }}>{engine.id}</td>
                  <td style={{ ...tdStyle }}>
                    <div style={{ display: 'inline-block', padding: '4px 8px', backgroundColor: `rgba(var(--${engine.status}-rgb), 0.1)`, border: `1px solid ${colorVar}`, color: colorVar, fontSize: '10px', letterSpacing: '1px', borderRadius: '2px' }}>
                      {engine.status === 'red' ? 'CRITICAL' : engine.status === 'amber' ? 'WATCH' : 'NOMINAL'}
                    </div>
                  </td>
                  <td style={{ ...tdStyle, color: colorVar, fontWeight: 500, fontSize: '16px' }}>{engine.rul.toFixed(1)}</td>
                  <td style={{ ...tdStyle, color: 'var(--ink2)' }}>{engine.conservativeRul.toFixed(1)}</td>
                  <td style={{ ...tdStyle }}>{engine.uncertainty.toFixed(1)}</td>
                  <td style={{ ...tdStyle, color: 'var(--ink3)' }}>{engine.trueRul.toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SystemView() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/api/metrics`)
      .then(res => res.json())
      .then(data => {
        setMetrics(data);
        setLoading(false);
      })
      .catch(err => console.error("Failed to fetch metrics:", err));
  }, []);

  if (loading) return <div style={{ padding: '32px', fontFamily: '"IBM Plex Mono"' }}>Loading metrics...</div>;
  if (!metrics) return <div style={{ padding: '32px', fontFamily: '"IBM Plex Mono"' }}>No metrics available.</div>;

  const MetricCard = ({ title, baseline, advanced, lowerIsBetter = true }) => {
    const imp = baseline ? ((baseline - advanced) / baseline * 100) : 0;
    const isImproved = lowerIsBetter ? imp > 0 : imp < 0;
    const absImp = Math.abs(imp).toFixed(1);
    
    return (
      <div style={{ padding: '24px', border: '1px solid var(--rule)', backgroundColor: 'var(--panel)' }}>
        <div style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '11px', color: 'var(--ink3)', letterSpacing: '1px', textTransform: 'uppercase', marginBottom: '16px' }}>
          {title}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '12px' }}>
          <div>
            <div style={{ fontSize: '11px', color: 'var(--ink3)', marginBottom: '4px' }}>Advanced CNN-LSTM</div>
            <div style={{ fontSize: '28px', fontFamily: '"IBM Plex Mono", monospace', fontWeight: 500 }}>
              {advanced?.toFixed(2) || 'N/A'}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '11px', color: 'var(--ink3)', marginBottom: '4px' }}>Baseline</div>
            <div style={{ fontSize: '16px', fontFamily: '"IBM Plex Mono", monospace', color: 'var(--ink2)' }}>
              {baseline?.toFixed(2) || 'N/A'}
            </div>
          </div>
        </div>
        {baseline && advanced && (
          <div style={{ 
            fontSize: '11px', 
            fontFamily: '"IBM Plex Mono", monospace', 
            color: isImproved ? 'var(--green)' : 'var(--red)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <span>{isImproved ? '↓' : '↑'}</span>
            <span>{absImp}% vs Baseline</span>
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={{ backgroundColor: 'var(--paper)', flex: 1, display: 'flex', flexDirection: 'column', padding: '32px' }}>
      <h2 className="section-header" style={{ marginBottom: '32px' }}>System Performance Metrics</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '48px' }}>
        <MetricCard 
          title="Root Mean Square Error (RMSE)" 
          baseline={metrics.baseline['RMSE']} 
          advanced={metrics.advanced['RMSE']} 
        />
        <MetricCard 
          title="NASA Asymmetric Score" 
          baseline={metrics.baseline['NASA Score']} 
          advanced={metrics.advanced['NASA Score']} 
        />
        <MetricCard 
          title="Mean Absolute Error (MAE)" 
          baseline={metrics.baseline['MAE']} 
          advanced={metrics.advanced['MAE']} 
        />
      </div>

      <h2 className="section-header" style={{ marginBottom: '24px' }}>Model Configuration</h2>
      <div style={{ 
        backgroundColor: 'var(--panel)', 
        border: '1px solid var(--rule)',
        padding: '24px',
        fontFamily: '"IBM Plex Mono", monospace',
        fontSize: '12px',
        color: 'var(--ink2)',
        lineHeight: 1.6
      }}>
        <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
          {JSON.stringify(metrics.config || { note: "No config.json found in model directory" }, null, 2)}
        </pre>
      </div>
    </div>
  );
}

// === Loading Indicator ===
function LoadingIndicator() {
  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center', 
      height: '100vh',
      fontFamily: '"IBM Plex Mono", monospace',
      fontSize: '11px',
      color: 'var(--ink3)',
      letterSpacing: '2px',
      textTransform: 'uppercase',
    }}>
      Loading fleet data…
    </div>
  );
}

// === Error Indicator ===
function ErrorIndicator({ message }) {
  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center', 
      height: '100vh',
      flexDirection: 'column',
      gap: '12px',
    }}>
      <div style={{ 
        fontFamily: '"IBM Plex Mono", monospace',
        fontSize: '11px',
        color: 'var(--red)',
        letterSpacing: '2px',
        textTransform: 'uppercase',
      }}>
        Connection Error
      </div>
      <div style={{ 
        fontFamily: '"IBM Plex Mono", monospace',
        fontSize: '9px',
        color: 'var(--ink3)',
        maxWidth: '400px',
        textAlign: 'center',
        lineHeight: '1.6',
      }}>
        {message || 'Unable to connect to the backend API. Ensure the FastAPI server is running on port 8000.'}
      </div>
    </div>
  );
}

// === Main App Component ===
function App() {
  const [fleet, setFleet] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [engineDetail, setEngineDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentView, setCurrentView] = useState('dashboard'); // 'dashboard', 'fleet', 'system'

  // Fetch fleet on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/fleet`)
      .then(res => res.json())
      .then(data => {
        setFleet(data.fleet);
        setLoading(false);
        // Default: select first critical/amber engine, or first engine
        const urgentIdx = data.fleet.findIndex(e => e.status === 'red');
        const firstIndex = urgentIdx >= 0 ? data.fleet[urgentIdx].index : data.fleet[0]?.index || 0;
        setSelectedIndex(firstIndex);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Fetch engine detail when selection changes
  useEffect(() => {
    if (fleet.length === 0) return;
    fetch(`${API_BASE}/api/engine/${selectedIndex}`)
      .then(res => res.json())
      .then(data => setEngineDetail(data))
      .catch(err => console.error('Failed to fetch engine detail:', err));
  }, [selectedIndex, fleet]);

  if (loading) return <LoadingIndicator />;
  if (error) return <ErrorIndicator message={error} />;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <NavBar currentView={currentView} setView={setCurrentView} />
      
      {currentView === 'dashboard' && (
        <DashboardView 
          fleet={fleet} 
          selectedIndex={selectedIndex} 
          setSelectedIndex={setSelectedIndex} 
          engineDetail={engineDetail} 
        />
      )}

      {currentView === 'fleet' && (
        <FleetView fleet={fleet} />
      )}

      {currentView === 'system' && (
        <SystemView />
      )}
      
    </div>
  );
}

export default App;
