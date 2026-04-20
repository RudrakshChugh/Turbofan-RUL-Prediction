import numpy as np

def generate_maintenance_alerts(mean_preds, std_preds, engine_ids, threshold=15):
    print("\n--- Maintenance Decision Logic ---")
    alerts_triggered = 0
    decisions = []
    
    for i, (mean, std, e_id) in enumerate(zip(mean_preds, std_preds, engine_ids)):
        conservative_rul = mean - (1.5 * std)
        
        if conservative_rul <= threshold:
            status = "CRITICAL: Schedule Maintenance IMMEDIATELY"
            alerts_triggered += 1
        elif mean <= threshold * 2:
            status = "WARNING: Prepare for upcoming maintenance"
        else:
            status = "HEALTHY"
            
        decision = {
            'Engine_ID': e_id,
            'Pred_RUL': mean,
            'Uncertainty_Std': std,
            'Conservative_RUL': conservative_rul,
            'Status': status
        }
        decisions.append(decision)
        
        if i < 10 or status.startswith("CRITICAL"):
            print(f"Engine {e_id} | Mean RUL: {mean:.1f} | Conservative RUL: {conservative_rul:.1f} --> {status}")
            
    print(f"\nTotal Engines Evaluated: {len(engine_ids)}")
    print(f"Total Immediate Maintenance Alerts: {alerts_triggered}")
    
    return decisions
