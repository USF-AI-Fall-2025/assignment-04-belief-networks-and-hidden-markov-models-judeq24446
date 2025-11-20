from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

def build_alarm_model() -> DiscreteBayesianNetwork:
    alarm_model = DiscreteBayesianNetwork(
        [
            ("Burglary", "Alarm"),
            ("Earthquake", "Alarm"),
            ("Alarm", "JohnCalls"),
            ("Alarm", "MaryCalls"),
        ]
    )

    cpd_burglary = TabularCPD(
        variable="Burglary", variable_card=2, values=[[0.999], [0.001]],
        state_names={"Burglary":['no','yes']},
    )
    cpd_earthquake = TabularCPD(
        variable="Earthquake", variable_card=2, values=[[0.998], [0.002]],
        state_names={"Earthquake":["no","yes"]},
    )
    cpd_alarm = TabularCPD(
        variable="Alarm",
        variable_card=2,
        values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
        evidence=["Burglary", "Earthquake"],
        evidence_card=[2, 2],
        state_names={"Burglary":['no','yes'], "Earthquake":['no','yes'], "Alarm":['no','yes']},
    )
    cpd_johncalls = TabularCPD(
        variable="JohnCalls",
        variable_card=2,
        values=[[0.95, 0.1], [0.05, 0.9]],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={"Alarm":['no','yes'], "JohnCalls":['no', 'yes']},
    )
    cpd_marycalls = TabularCPD(
        variable="MaryCalls",
        variable_card=2,
        values=[[0.99, 0.3], [0.01, 0.7]],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={"Alarm":['no','yes'], "MaryCalls":['no', 'yes']},
    )

    # Associating the parameters with the model structure
    alarm_model.add_cpds(
        cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)
    alarm_model.check_model()
    return alarm_model

def main():
    alarm_model = build_alarm_model()
    alarm_infer = VariableElimination(alarm_model)

    #print(alarm_infer.query(variables=["JohnCalls"],evidence={"Earthquake":"yes"}))
    #
    #the probability of Mary Calling given that John called
    # 4% chance Mary calls if John calls
    q1 = alarm_infer.query(variables=["MaryCalls"],evidence={"JohnCalls":"yes"})
    print("P(MaryCalls | JohnCalls) = 0.04")
    print(q1)

    #the probability of both John and Mary calling given that the alarm has gone off
    # 63% chance both John and Mary call if the alarm goes off
    q2 = alarm_infer.query(variables=["JohnCalls", "MaryCalls"],evidence={"Alarm":"yes"})
    print("\nP(JohnCalls, MaryCalls | Alarm) = 0.63")
    print(q2)

    #the probability of the alarm going off given that Mary called
    # 15.01% chance the alarm went off if Mary calls
    q3 = alarm_infer.query(variables=["Alarm"],evidence={"MaryCalls":"yes"})
    print("\nP(Alarm | MaryCalls) = 0.1501")
    print(q3)

if __name__ == "__main__":
    main()
