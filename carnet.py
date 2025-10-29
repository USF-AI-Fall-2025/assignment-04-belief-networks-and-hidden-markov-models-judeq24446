from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

def build_car_model() -> DiscreteBayesianNetwork:
    car_model = DiscreteBayesianNetwork(
        [
            ("Battery", "Radio"),
            ("Battery", "Ignition"),
            ("Ignition","Starts"),
            ("Gas","Starts"),
            ("Starts","Moves"),
    ])

    cpd_battery = TabularCPD(
        variable="Battery", variable_card=2, values=[[0.70], [0.30]],
        state_names={"Battery":['Works',"Doesn't work"]},
    )

    cpd_gas = TabularCPD(
        variable="Gas", variable_card=2, values=[[0.40], [0.60]],
        state_names={"Gas":['Full',"Empty"]},
    )

    cpd_radio = TabularCPD(
        variable=  "Radio", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Radio": ["turns on", "Doesn't turn on"],
                    "Battery": ['Works',"Doesn't work"]}
    )

    cpd_ignition = TabularCPD(
        variable=  "Ignition", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Ignition": ["Works", "Doesn't work"],
                    "Battery": ['Works',"Doesn't work"]}
    )

    cpd_starts = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
        evidence=["Ignition", "Gas"],
        evidence_card=[2, 2],
        state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"]},
    )

    cpd_moves = TabularCPD(
        variable="Moves", variable_card=2,
        values=[[0.8, 0.01],[0.2, 0.99]],
        evidence=["Starts"],
        evidence_card=[2],
        state_names={"Moves": ["yes", "no"],
                    "Starts": ['yes', 'no'] }
    )


    # Associating the parameters with the model structure
    car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves)
    car_model.check_model()
    return car_model

def build_car_model_with_key() -> DiscreteBayesianNetwork:
    car_model = DiscreteBayesianNetwork(
        [
            ("Battery", "Radio"),
            ("Battery", "Ignition"),
            ("Ignition","Starts"),
            ("Gas","Starts"),
            ("KeyPresent", "Starts"),
            ("Starts","Moves"),
    ])

    cpd_battery = TabularCPD(
        variable="Battery", variable_card=2, values=[[0.70], [0.30]],
        state_names={"Battery":['Works',"Doesn't work"]},
    )

    cpd_gas = TabularCPD(
        variable="Gas", variable_card=2, values=[[0.40], [0.60]],
        state_names={"Gas":['Full',"Empty"]},
    )

    cpd_radio = TabularCPD(
        variable=  "Radio", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Radio": ["turns on", "Doesn't turn on"],
                    "Battery": ['Works',"Doesn't work"]}
    )

    cpd_ignition = TabularCPD(
        variable=  "Ignition", variable_card=2,
        values=[[0.75, 0.01],[0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Ignition": ["Works", "Doesn't work"],
                    "Battery": ['Works',"Doesn't work"]}
    )

    cpd_key = TabularCPD(
        variable="KeyPresent", variable_card=2, 
        values=[[0.7], [0.3]],
        state_names={"KeyPresent":['yes','no']},
    )

    # list comprehension to create the CPT for Starts | Ignition, Gas, KeyPresent
    yes = []
    # Populate the CPT for Starts | Ignition, Gas, KeyPresent
    for gas in ['Full', 'Empty']:
        for ignition in ['Works', "Doesn't work"]:
            for key in ['yes', 'no']:
                if gas == 'Full' and ignition == 'Works' and key == 'yes':
                    yes.append(0.99)
                else:
                    yes.append(0.01)
    no = [1 - p for p in yes]

    cpd_starts = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[yes, no],
        evidence=["Ignition", "Gas", "KeyPresent"],
        evidence_card=[2, 2, 2],
        state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent": ['yes', 'no']},
    )

    cpd_moves = TabularCPD(
        variable="Moves", variable_card=2,
        values=[[0.8, 0.01],[0.2, 0.99]],
        evidence=["Starts"],
        evidence_card=[2],
        state_names={"Moves": ["yes", "no"],
                    "Starts": ['yes', 'no'] }
    )


    # Associating the parameters with the model structure
    car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key)
    car_model.check_model()
    return car_model

def step2():
    car_model = build_car_model()
    car_infer = VariableElimination(car_model)

    print("Step 2 Query Results:")
    # Given that the car will not move, what is the probability that the battery is not working?
    # 35.9% chance the battery is not working
    print("\nP(!battery | !moves) = 0.359")
    print(car_infer.query(variables=["Battery"],evidence={"Moves":"no"}))

    # Given that the radio is not working, what is the probability that the car will not start?
    # 86.9% chance the car will not start
    print("\nP(!starts | !radio) = 0.869")
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"Doesn't turn on"}))

    # Given that the battery is working, does the probability of the radio working change if we discover
    # that the car has gas in it?
    # The probability the radio works remains 75% whether or not the car has gas in it
    print("\nP(radio | battery), P(radio | battery, gas)")
    print("The probability the radio works remains 75% whether or not the car has gas in it")
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works"}))
    print(car_infer.query(variables=["Radio"],evidence={"Battery":"Works", "Gas":"Full"}))

    # Given that the car doesn't move, how does the probability of the ignition failing change if we
    # observe that the car does not have gas in it?
    # The probability of the ignition failing decreases from 56.66% to 48.22% if we observe that the car does not have gas in it
    print("\nP(!ignition | !moves), P(!ignition | !moves, !gas)")
    print("The probability of the ignition failing decreases from 56.66% to 48.22% if we observe that the car does not have gas in it")
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no"}))
    print(car_infer.query(variables=["Ignition"],evidence={"Moves":"no", "Gas":"Empty"}))

    # What is the probability that the car starts if the radio works and it has gas in it?
    # 72.12% chance the car starts if the radio works and it has gas in it
    print("\nP(starts | radio, gas) = 0.7212")
    print(car_infer.query(variables=["Starts"],evidence={"Radio":"turns on", "Gas":"Full"}))

def step3():
    car_model_with_key = build_car_model_with_key()
    car_infer_with_key = VariableElimination(car_model_with_key)

    print("\nStep 3 Query Results:")

    # helper to normalize pgmpy query results and extract the probability for the 'yes' state
    def get_yes_prob(result, var_name="Starts", yes_label="yes"):
        factor = result[var_name] if isinstance(result, dict) else result
        sn = getattr(factor, "state_names", None)
        idx = 0
        if isinstance(sn, dict) and var_name in sn:
            try:
                idx = sn[var_name].index(yes_label)
            except ValueError:
                idx = 0
        return float(factor.values[idx])

    # different queries showing the effect of the key being present on the probability of the car starting
    q1 = car_infer_with_key.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Works", "KeyPresent": "yes"})
    print(f"P(starts | gas, ignition, keyPresent) = {get_yes_prob(q1, 'Starts')}" )

    q2 = car_infer_with_key.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Doesn't work", "KeyPresent": "yes"})
    print(f"P(starts | gas, !ignition, keyPresent) = {get_yes_prob(q2, 'Starts')}")

    q3 = car_infer_with_key.query(variables=["Starts"], evidence={"Gas": "Empty", "Ignition": "Works", "KeyPresent": "yes"})
    print(f"P(starts | !gas, ignition, keyPresent) = {get_yes_prob(q3, 'Starts')}")

    q4 = car_infer_with_key.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Works", "KeyPresent": "no"})
    print(f"P(starts | gas, ignition, !keyPresent) = {get_yes_prob(q4, 'Starts')}")

    q5 = car_infer_with_key.query(variables=["Starts"], evidence={"Gas": "Empty", "Ignition": "Doesn't work", "KeyPresent": "yes"})
    print(f"P(starts | !gas, !ignition, keyPresent) = {get_yes_prob(q5, 'Starts')}")

    q6 = car_infer_with_key.query(variables=["Starts"], evidence={"Gas": "Empty", "Ignition": "Works", "KeyPresent": "no"})
    print(f"P(starts | !gas, ignition, !keyPresent) = {get_yes_prob(q6, 'Starts')}")

    q7 = car_infer_with_key.query(variables=["Starts"], evidence={"Gas": "Full", "Ignition": "Doesn't work", "KeyPresent": "no"})
    print(f"P(starts | gas, !ignition, !keyPresent) = {get_yes_prob(q7, 'Starts')}")

    q8 = car_infer_with_key.query(variables=["Starts"], evidence={"Gas": "Empty", "Ignition": "Doesn't work", "KeyPresent": "no"})
    print(f"P(starts | !gas, !ignition, !keyPresent) = {get_yes_prob(q8, 'Starts')}")
if __name__ == "__main__":
    step2()
    print()
    step3()