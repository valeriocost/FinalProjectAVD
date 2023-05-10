import json

if __name__ == "__main__":
    
  
    # Opening JSON file
    f = open('simulation_results_BASELINE.json')
    
    data = json.load(f)
    result_scenarios = {i:{} for i in range(len(data['_checkpoint']['records']))}

    # Iterating through the json list
    for i, scenario in enumerate(data['_checkpoint']['records']):
        for infraction, values in scenario['infractions'].items():
            result_scenarios[i][infraction] = len(values)
    
    print(result_scenarios)
    # Closing file
    f.close()