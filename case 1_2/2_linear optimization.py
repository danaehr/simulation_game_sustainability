########################################################################################
# Project Pipeline
########################################################################################

# Step 1:
# 0_exploration_and_cleaning.py
# -> Data exploration and preprocessing
# -> Cleaning missing values and formatting time series data

# Step 2:
# 1_predict_prices.py
# cobalt_xg_boost.py
# lithium_xg_boost.py
#
# -> Forecasting future material prices until 2030
# -> Uses SARIMA / XGBoost depending on material
# -> Output stored as CSV files in:
#    ./data/2_predicted/

# Step 3:
# 2_linear_optimization.py
# -> Loads predicted supplier prices
# -> Optimizes supplier selection under:
#       - budget constraints
#       - sustainability constraints
#       - material demand constraints
# -> Maximizes the number of produced cars

########################################################################################


# Imports
import sys
from pulp import LpMaximize, LpProblem, LpVariable
import matplotlib.pyplot as plt
import pandas as pd


def calculate_supplier_selection():
    ### Initialize model
    model = LpProblem(name='Car Production', sense=LpMaximize)

    ########################################################################################
    ###################################### Variables #######################################
    ########################################################################################

    ### Represents the amount of steel provided by a company. The variable is continuous,
    ### meaning that it can take any real value greater than zero
    steel_amount_east_metal = LpVariable(name='steel_amount_east_metal', lowBound=0, cat='Continuous')
    steel_amount_sakura = LpVariable(name='steel_amount_sakura', lowBound=0, cat='Continuous')
    steel_amount_black_forest = LpVariable(name='steel_amount_black_forest', lowBound=0, cat='Continuous')

    ### Binary variable that indicates whether the companies steel is used or not. If the variable equals 1,
    ### it means that the company's steel is used. If it equals 0, it means that the company does not provide steel
    use_steel_east_metal = LpVariable(name='use_steel_east_metal', lowBound=0, cat='Binary')
    use_steel_sakura = LpVariable(name='use_steel_sakura', lowBound=0, cat='Binary')
    use_steel_black_forest = LpVariable(name='use_steel_black_forest', lowBound=0, cat='Binary')

    ### Represents the amount of aluminium provided by a company. The variable is continuous,
    ### meaning that it can take any real value greater than zero
    aluminium_amount_rising_sun = LpVariable(name='aluminium_amount_rising_sun', lowBound=0, cat='Continuous')
    aluminium_amount_rhine_metal = LpVariable(name='aluminium_amount_rhine_metal', lowBound=0, cat='Continuous')
    aluminium_amount_fjordlight = LpVariable(name='aluminium_amount_fjordlight', lowBound=0, cat='Continuous')

    ### Binary variable that indicates whether the company's aluminium is used or not. If the variable equals 1,
    ### it means that the company's aluminium is used. If it equals 0, it means that the company does not provide aluminium
    use_aluminium_rising_sun = LpVariable(name='use_aluminium_rising_sun', lowBound=0, cat='Binary')
    use_aluminium_rhine_metal = LpVariable(name='use_aluminium_rhine_metal', lowBound=0, cat='Binary')
    use_aluminium_fjordlight = LpVariable(name='use_aluminium_fjordlight', lowBound=0, cat='Binary')

    ### Represents the amount of microchips provided by a company. The variable is continuous,
    ### meaning that it can take any real value greater than zero
    microchips_amount_skytech = LpVariable(name='microchips_amount_skytech', lowBound=0, cat='Continuous')
    microchips_amount_sic = LpVariable(name='microchips_amount_sic', lowBound=0, cat='Continuous')
    microchips_amount_nexgen = LpVariable(name='microchips_amount_nexgen', lowBound=0, cat='Continuous')

    ### Binary variable that indicates whether the company's microchips are used or not. If the variable equals 1,
    ### it means that the company's microchips are used. If it equals 0, it means that the company does not provide microchips
    use_microchips_skytech = LpVariable(name='use_microchips_skytech', lowBound=0, cat='Binary')
    use_microchips_sic = LpVariable(name='use_microchips_sic', lowBound=0, cat='Binary')
    use_microchips_nexgen = LpVariable(name='use_microchips_nexgen', lowBound=0, cat='Binary')

    ### Represents the amount of cobalt provided by a company. The variable is continuous,
    ### meaning that it can take any real value greater than zero
    cobalt_amount_congo_cobalt = LpVariable(name='cobalt_amount_congo_cobalt', lowBound=0, cat='Continuous')
    cobalt_amount_auric_cobalt = LpVariable(name='cobalt_amount_auric_cobalt', lowBound=0, cat='Continuous')
    cobalt_amount_ncc = LpVariable(name='cobalt_amount_ncc', lowBound=0, cat='Continuous')

    ### Binary variable that indicates whether the company's cobalt is used or not. If the variable equals 1,
    ### it means that the company's cobalt is used. If it equals 0, it means that the company does not provide cobalt
    use_cobalt_congo_cobalt = LpVariable(name='use_cobalt_congo_cobalt', lowBound=0, cat='Binary')
    use_cobalt_auric_cobalt = LpVariable(name='use_cobalt_auric_cobalt', lowBound=0, cat='Binary')
    use_cobalt_ncc = LpVariable(name='use_cobalt_ncc', lowBound=0, cat='Binary')

    ### Represents the amount of lithium provided by a company. The variable is continuous,
    ### meaning that it can take any real value greater than zero
    lithium_amount_sollith = LpVariable(name='lithium_amount_sollith', lowBound=0, cat='Continuous')
    lithium_amount_litio_andes = LpVariable(name='lithium_amount_litio_andes', lowBound=0, cat='Continuous')
    lithium_amount_lithiumoz = LpVariable(name='lithium_amount_lithiumoz', lowBound=0, cat='Continuous')

    ### Binary variable that indicates whether the company's lithium is used or not. If the variable equals 1,
    ### it means that the company's lithium is used. If it equals 0, it means that the company does not provide lithium
    use_lithium_chile_sollith = LpVariable(name='use_lithium_chile_sollith', lowBound=0, cat='Binary')
    use_lithium_litio_andes = LpVariable(name='use_lithium_litio_andes', lowBound=0, cat='Binary')
    use_lithium_lithiumoz = LpVariable(name='use_lithium_lithiumoz', lowBound=0, cat='Binary')

    ### Number of cars produced
    car_amount = LpVariable(name='car_amount', lowBound=0, cat='Integer')

    ########################################################################################
    ###################################### Constants #######################################
    ########################################################################################
    # Load predicted prices from CSV files
    # Use the LAST predicted value from each supplier

    # Steel
    steel_df = pd.read_csv("./data/2_predicted/final_predict_steel_price.csv", sep=";")

    steel_price_east_metal = steel_df["East Metal Co."].iloc[-1]
    steel_price_sakura = steel_df["Sakura Steelworks"].iloc[-1]
    steel_price_black_forest = steel_df["Black Forest Steel Co."].iloc[-1]

    # Aluminium
    aluminium_df = pd.read_csv("./data/2_predicted/Aluminium_Price_2026_predicted.csv")

    aluminium_price_rising_sun = aluminium_df["Rising Sun Aluminum"].iloc[-1]
    aluminium_price_rhine_metal = aluminium_df["RhineMetal Aluminum"].iloc[-1]
    aluminium_price_fjordlight = aluminium_df["Fjordlight Aluminum"].iloc[-1]

    # Microchips
    microchips_df = pd.read_csv("./data/2_predicted/Microchips_Price_2026_predicted.csv")

    microchips_price_skytech = microchips_df["Skytech Microelectronics Ltd."].iloc[-1]
    microchips_price_sic = microchips_df["Silicon Innovations Corporation"].iloc[-1]
    microchips_price_nexgen = microchips_df["NexGen Microsystems GmbH"].iloc[-1]

    # Cobalt
    cobalt_df = pd.read_csv("./data/2_predicted/cobalt_forecast_xgboost.csv")

    cobalt_price_congo_cobalt = cobalt_df["Congo Cobalt"].iloc[-1]
    cobalt_price_auric_cobalt = cobalt_df["Auric Cobalt"].iloc[-1]
    cobalt_price_ncc = cobalt_df["Northern Cobalt Corporation"].iloc[-1]

    # Lithium
    lithium_df = pd.read_csv("./data/2_predicted/lithium_forecast_xgboost.csv")

    lithium_price_sollith = lithium_df["SolLith"].iloc[-1]
    lithium_price_litio_andes = lithium_df["LitioAndes"].iloc[-1]
    lithium_price_lithiumoz = lithium_df["LithiumOz"].iloc[-1]

    ########################################################################################
    ######################## Sustainability / ESG Scores ###################################
    ########################################################################################

    # Lower score = better supplier
    # Final score combines:
    # - CO2 emissions
    # - ethical / social sustainability aspects
    #
    # Ethics is intentionally weighted stronger than CO2.
    # This includes:
    # - human rights
    # - labor conditions
    # - child labor
    # - water scarcity
    # - environmental destruction affecting local communities

    co2_weight = 1
    ethics_weight = 15

    # =============================================================================
    # STEEL
    # =============================================================================

    # East Metal:
    # - labor exploitation
    # - corruption concerns
    # - weaker labor standards
    # - high CO2 emissions
    steel_score_east_metal = (
        co2_weight * 2.6 +
        ethics_weight * 0.80
    )

    # Sakura:
    # - strong regulation
    # - sustainable production
    # - lower emissions
    steel_score_sakura = (
        co2_weight * 1.91 +
        ethics_weight * 0.20
    )

    # Black Forest:
    # - strict regulation
    # - CO2 certificates
    # - strongest sustainability focus
    steel_score_black_forest = (
        co2_weight * 1.81 +
        ethics_weight * 0.10
    )

    # =============================================================================
    # ALUMINIUM
    # =============================================================================

    # Rising Sun:
    # - destruction of agricultural land
    # - water damage
    # - health risks from dust
    # - high CO2 emissions
    aluminium_score_rising_sun = (
        co2_weight * 12.5 +
        ethics_weight * 0.90
    )

    # RhineMetal:
    # - strong regulation
    # - moderate emissions
    # - sustainability investments
    aluminium_score_rhine_metal = (
        co2_weight * 5.5 +
        ethics_weight * 0.25
    )

    # Fjordlight:
    # - low emissions
    # - few social concerns
    # - advanced sustainable production
    aluminium_score_fjordlight = (
        co2_weight * 2.0 +
        ethics_weight * 0.10
    )

    # =============================================================================
    # MICROCHIPS
    # =============================================================================

    # Skytech:
    # - unpaid overtime
    # - surveillance concerns
    # - problematic wages
    microchips_score_skytech = (
        co2_weight * 58 +
        ethics_weight * 1.00
    )

    # SIC:
    # - strong regulation
    # - high automation
    # - moderate sustainability
    microchips_score_sic = (
        co2_weight * 34 +
        ethics_weight * 0.30
    )

    # NexGen:
    # - strong worker protection
    # - sustainable production
    # - low emissions
    microchips_score_nexgen = (
        co2_weight * 29 +
        ethics_weight * 0.10
    )

    # =============================================================================
    # COBALT
    # =============================================================================

    # CongoCobalt:
    # - child labor
    # - human rights violations
    # - very problematic mining conditions
    cobalt_score_congo_cobalt = (
        co2_weight * 42 +
        ethics_weight * 1.00
    )

    # Auric:
    # - displacement of local communities
    # - otherwise regulated production
    cobalt_score_auric_cobalt = (
        co2_weight * 38 +
        ethics_weight * 0.50
    )

    # NCC:
    # - strict regulation
    # - sustainable production
    # - low emissions
    cobalt_score_ncc = (
        co2_weight * 5.5 +
        ethics_weight * 0.10
    )

    # =============================================================================
    # LITHIUM
    # =============================================================================

    # SolLith:
    # - extremely high water usage
    # - severe water scarcity concerns
    # - ecosystem damage
    lithium_score_sollith = (
        co2_weight * 15 +
        ethics_weight * 0.95
    )

    # LitioAndes:
    # - lower water use
    # - contamination risks remain
    lithium_score_litio_andes = (
        co2_weight * 16 +
        ethics_weight * 0.75
    )

    # LithiumOz:
    # - strong regulation
    # - environmental protection measures
    lithium_score_lithiumoz = (
        co2_weight * 18 +
        ethics_weight * 0.20
    )
        
    ### Material demand per car
    steel_demand = 600
    aluminium_demand = 90
    microchips_demand = 1200
    cobalt_demand = 15
    lithium_demand = 14

    ### Production budget
    budget = 2500000
    

    ### Helpers
    M = sys.maxsize



    ########################################################################################
    ##################################### Constraints ######################################
    ########################################################################################

    ### Calculate the amount of steel required per car by summing
    ### the steel amounts from three different sources
    steel_per_car = car_amount == steel_amount_east_metal * (1 / steel_demand) \
                    + steel_amount_black_forest * (1 / steel_demand) \
                    + steel_amount_sakura * (1 / steel_demand)
    model += (steel_per_car, "steel_per_car")

    ### Enforce a constraint that allows only one steel supplier to be used for the optimization by setting the
    ### upper bound of the steel amount for each supplier to be M (a very large number) times their respective
    ### binary decision variables. This ensures that the amount of steel from a particular supplier is zero if
    ### the corresponding binary variable is not selected
    one_steel_east_metal = steel_amount_east_metal <= M * use_steel_east_metal
    model += (one_steel_east_metal, "one_steel_east_metal")
    one_steel_sakura = steel_amount_sakura <= M * use_steel_sakura
    model += (one_steel_sakura, "one_steel_sakura")
    one_steel_black_forest = steel_amount_black_forest <= M * use_steel_black_forest
    model += (one_steel_black_forest, "one_steel_black_forest")
    only_one_steel = (use_steel_east_metal + use_steel_black_forest + use_steel_sakura <= 1) #adjusted here beacause else code doesnt run
    model += only_one_steel

    ### Calculate the amount of aluminium required per car by summing
    ### the steel amounts from three different sources
    aluminium_per_car = car_amount == aluminium_amount_rising_sun * (1 / aluminium_demand) \
                        + aluminium_amount_rhine_metal * (1 / aluminium_demand) \
                        + aluminium_amount_fjordlight * (1 / aluminium_demand)
    model += (aluminium_per_car, "aluminium_per_car")

    ### Enforce a constraint that allows only one aluminium supplier to be used for the optimization by setting the
    ### upper bound of the aluminium amount for each supplier to be M (a very large number) times their respective
    ### binary decision variables. This ensures that the amount of aluminium from a particular supplier is zero if
    ### the corresponding binary variable is not selected
    one_aluminium_rising_sun = aluminium_amount_rising_sun <= M * use_aluminium_rising_sun
    model += (one_aluminium_rising_sun, "one_aluminium_rising_sun ")
    one_aluminium_rhine_metal = aluminium_amount_rhine_metal <= M * use_aluminium_rhine_metal
    model += (one_aluminium_rhine_metal, "one_aluminium_rhine_metal")
    one_aluminium_fjordlight = aluminium_amount_fjordlight <= M * use_aluminium_fjordlight
    model += (one_aluminium_fjordlight, "one_aluminium_fjordlight")
    only_one_aluminium = (use_aluminium_rising_sun + use_aluminium_rhine_metal + use_aluminium_fjordlight <= 1) #also adjusted here
    model += only_one_aluminium

    ### Calculate the amount of microchips required per car by summing
    ### the steel amounts from three different sources
    microchips_per_car = car_amount == microchips_amount_skytech * (1 / microchips_demand) \
                         + microchips_amount_sic * (1 / microchips_demand) \
                         + microchips_amount_nexgen * (1 / microchips_demand)
    model += (microchips_per_car, "microchips_per_car")

    ### Enforce a constraint that allows only one microchip supplier to be used for the optimization by setting the
    ### upper bound of the microchip amount for each supplier to be M (a very large number) times their respective
    ### binary decision variables. This ensures that the amount of microchips from a particular supplier is zero if
    ### the corresponding binary variable is not selected
    one_microchips_skytech = microchips_amount_skytech <= M * use_microchips_skytech
    model += (one_microchips_skytech, "one_microchips_skytech")
    one_microchips_sic = microchips_amount_sic <= M * use_microchips_sic
    model += (one_microchips_sic, "one_microchips_sic")
    one_microchips_nexgen = microchips_amount_nexgen <= M * use_microchips_nexgen
    model += (one_microchips_nexgen, "one_microchips_nexgen")
    only_one_microchips = (use_microchips_skytech + use_microchips_sic + use_microchips_nexgen <= 1) #here too
    model += only_one_microchips

    #### Calculate the amount of cobalt required per car by summing
    ### the steel amounts from three different sources
    cobalt_per_car = car_amount == cobalt_amount_congo_cobalt * (1 / cobalt_demand) \
                     + cobalt_amount_auric_cobalt * (1 / cobalt_demand) \
                     + cobalt_amount_ncc * (1 / cobalt_demand)
    model += (cobalt_per_car, "cobalt_per_car")

    ### Enforce a constraint that allows only one cobalt supplier to be used for the optimization by setting the
    ### upper bound of the cobalt amount for each supplier to be M (a very large number) times their respective
    ### binary decision variables. This ensures that the amount of cobalt from a particular supplier is zero if
    ### the corresponding binary variable is not selected
    one_cobalt_congo_cobalt = cobalt_amount_congo_cobalt <= M * use_cobalt_congo_cobalt
    model += (one_cobalt_congo_cobalt, "one_cobalt_congo_cobalt")
    one_cobalt_auric_cobalt = cobalt_amount_auric_cobalt <= M * use_cobalt_auric_cobalt
    model += (one_cobalt_auric_cobalt, "one_cobalt_auric_cobalt")
    one_cobalt_ncc = cobalt_amount_ncc <= M * use_cobalt_ncc
    model += (one_cobalt_ncc, "one_cobalt_ncc")
    only_one_cobalt = (use_cobalt_congo_cobalt + use_cobalt_auric_cobalt + use_cobalt_ncc <= 1) #adjusted
    model += only_one_cobalt

    ### Calculate the amount of lithium required per car by summing
    ### the steel amounts from three different sources
    lithium_per_car = car_amount == lithium_amount_sollith * (1 / lithium_demand) \
                      + lithium_amount_litio_andes * (1 / lithium_demand) \
                      + lithium_amount_lithiumoz * (1 / lithium_demand)
    model += (lithium_per_car, "lithium_per_car")

    ### Enforce a constraint that allows only one lithium supplier to be used for the optimization by setting the
    ### upper bound of the lithium amount for each supplier to be M (a very large number) times their respective
    ### binary decision variables. This ensures that the amount of lithium from a particular supplier is zero if
    ### the corresponding binary variable is not selected
    one_lithium_sollith = lithium_amount_sollith <= M * use_lithium_chile_sollith
    model += (one_lithium_sollith, "one_lithium_sollith")
    one_lithium_litio_andes = lithium_amount_litio_andes <= M * use_lithium_litio_andes
    model += (one_lithium_litio_andes, "one_lithium_litio_andes")
    one_lithium_lithiumoz = lithium_amount_lithiumoz <= M * use_lithium_lithiumoz
    model += (one_lithium_lithiumoz, "one_lithium_lithiumoz")
    only_one_lithium = (use_lithium_chile_sollith + use_lithium_litio_andes + use_lithium_lithiumoz <= 1)
    model += only_one_lithium

    ### Enforce a budget constraint by calculating the total cost of all materials required for the car,
    ### and lithium, and ensuring that the cost does not exceed the available budget
    budget_constraint = budget >= \
                        (steel_amount_east_metal / 1000) * steel_price_east_metal \
                        + (steel_amount_sakura / 1000) * steel_price_sakura \
                        + (steel_amount_black_forest / 1000) * steel_price_black_forest \
                        + (aluminium_amount_rising_sun / 1000) * aluminium_price_rising_sun \
                        + (aluminium_amount_rhine_metal / 1000) * aluminium_price_rhine_metal \
                        + (aluminium_amount_fjordlight / 1000) * aluminium_price_fjordlight \
                        + (microchips_amount_skytech / 1500) * microchips_price_skytech \
                        + (microchips_amount_sic / 1500) * microchips_price_sic \
                        + (microchips_amount_nexgen / 1500) * microchips_price_nexgen \
                        + (cobalt_amount_congo_cobalt / 1000) * cobalt_price_congo_cobalt \
                        + (cobalt_amount_auric_cobalt / 1000) * cobalt_price_auric_cobalt \
                        + (cobalt_amount_ncc / 1000) * cobalt_price_ncc \
                        + (lithium_amount_sollith / 1000) * lithium_price_sollith \
                        + (lithium_amount_litio_andes / 1000) * lithium_price_litio_andes \
                        + (lithium_amount_lithiumoz / 1000) * lithium_price_lithiumoz
    
    # ------------------------------------------------------------
    # Sustainability penalty
    # Lower score = more sustainable supplier
    # ------------------------------------------------------------

    sustainability_penalty = (

        # Steel ($ per ton -> kg / 1000)
        (steel_amount_east_metal / 1000) * steel_score_east_metal +
        (steel_amount_sakura / 1000) * steel_score_sakura +
        (steel_amount_black_forest / 1000) * steel_score_black_forest +

        # Aluminium ($ per ton -> kg / 1000)
        (aluminium_amount_rising_sun / 1000) * aluminium_score_rising_sun +
        (aluminium_amount_rhine_metal / 1000) * aluminium_score_rhine_metal +
        (aluminium_amount_fjordlight / 1000) * aluminium_score_fjordlight +

        # Microchips ($ per 1500 units)
        (microchips_amount_skytech / 1500) * microchips_score_skytech +
        (microchips_amount_sic / 1500) * microchips_score_sic +
        (microchips_amount_nexgen / 1500) * microchips_score_nexgen +

        # Cobalt ($ per ton -> kg / 1000)
        (cobalt_amount_congo_cobalt / 1000) * cobalt_score_congo_cobalt +
        (cobalt_amount_auric_cobalt / 1000) * cobalt_score_auric_cobalt +
        (cobalt_amount_ncc / 1000) * cobalt_score_ncc +

        # Lithium ($ per ton -> kg / 1000)
        (lithium_amount_sollith / 1000) * lithium_score_sollith +
        (lithium_amount_litio_andes / 1000) * lithium_score_litio_andes +
        (lithium_amount_lithiumoz / 1000) * lithium_score_lithiumoz
    )

    model += (budget_constraint, "budget_constraint")

    ########################################################################################
    ##################################### Solve model ######################################
    ########################################################################################

    # Weight for sustainability influence
    lambda_weight = 0.005

    # Objective:
    # maximize car production
    # while minimizing sustainability penalty
    obj_func = car_amount - lambda_weight * sustainability_penalty

    model += obj_func
    # Solve model
    status = model.solve()

    ########################################################################################
    ################################### Output results #####################################
    ########################################################################################

    print("\n#################### Production ####################")
    print("### Number of cars produced ####")
    print("Number of cars produced: " + str(car_amount.value()))

    print("\n################## Materials ##################")
    print("### Steel")
    print("Amount of steel (East Metal Co.): " + str(steel_amount_east_metal.value()))
    print("Amount of steel (Sakura Steelworks): " + str(steel_amount_sakura.value()))
    print("Amount of steel (Black Forest Steel Co.): " + str(steel_amount_black_forest.value()))
    print("### Aluminium")
    print("Amount of aluminium (Rising Sun Aluminium): " + str(aluminium_amount_rising_sun.value()))
    print("Amount of aluminium (RhineMetal Aluminium): " + str(aluminium_amount_rhine_metal.value()))
    print("Amount of aluminium (Fjordlight Aluminium): " + str(aluminium_amount_fjordlight.value()))
    print("### Microchips")
    print("Amount of microchips (Skytech Microelectronics Ltd.): " + str(microchips_amount_skytech.value()))
    print("Amount of microchips (Silicon Innovations Corporation): " + str(microchips_amount_sic.value()))
    print("Amount of microchips (NexGen Microsystems GmbH): " + str(microchips_amount_nexgen.value()))
    print("### Cobalt")
    print("Amount of cobalt (Congo Cobalt): " + str(cobalt_amount_congo_cobalt.value()))
    print("Amount of cobalt (AuricCobalt): " + str(cobalt_amount_auric_cobalt.value()))
    print("Amount of cobalt (NCC): " + str(cobalt_amount_ncc.value()))
    print("### Lithium")
    print("Amount of lithium (SolLith): " + str(lithium_amount_sollith.value()))
    print("Amount of lithium (LitioAndes): " + str(lithium_amount_litio_andes.value()))
    print("Amount of lithium (LithiumOz): " + str(lithium_amount_lithiumoz.value()))

    print("\n################### Costs ###################")
    print("### Steel")
    print("Costs of steel (East Metal Co.): " + str((steel_amount_east_metal.value() / 1000) * steel_price_east_metal))
    print("Costs of steel (Sakura Steelworks): " + str((steel_amount_sakura.value() / 1000) * steel_price_sakura))
    print("Costs of steel (Black Forest Steel Co.): " + str((steel_amount_black_forest.value() / 1000) * steel_price_black_forest))

    print("### Aluminium")
    print("Costs of aluminium (Rising Sun Aluminium): " + str((aluminium_amount_rising_sun.value() / 1000) * aluminium_price_rising_sun))
    print("Costs of aluminium (RhineMetal Aluminium): " + str((aluminium_amount_rhine_metal.value() / 1000) * aluminium_price_rhine_metal))
    print("Costs of aluminium (Fjordlight Aluminium): " + str((aluminium_amount_fjordlight.value() / 1000) * aluminium_price_fjordlight))

    print("### Microchips")
    print("Costs of microchips (Skytech Microelectronics Ltd.): " + str((microchips_amount_skytech.value() / 1500) * microchips_price_skytech))
    print("Costs of microchips (Silicon Innovations Corporation): " + str((microchips_amount_sic.value() / 1500) * microchips_price_sic))
    print("Costs of microchips (NexGen Microsystems GmbH): " + str((microchips_amount_nexgen.value() / 1500) * microchips_price_nexgen))

    print("### Cobalt")
    print("Costs of cobalt (Congo Cobalt): " + str((cobalt_amount_congo_cobalt.value() / 1000) * cobalt_price_congo_cobalt))
    print("Costs of cobalt (AuricCobalt): " + str((cobalt_amount_auric_cobalt.value() / 1000) * cobalt_price_auric_cobalt))
    print("Costs of cobalt (NCC): " + str((cobalt_amount_ncc.value() / 1000) * cobalt_price_ncc))

    print("### Lithium")
    print("Costs of lithium (SolLith): " + str((lithium_amount_sollith.value() / 1000) * lithium_price_sollith))
    print("Costs of lithium (LitioAndes): " + str((lithium_amount_litio_andes.value() / 1000) * lithium_price_litio_andes))
    print("Costs of lithium (LithiumOz): " + str((lithium_amount_lithiumoz.value() / 1000) * lithium_price_lithiumoz))
         # --------------------------------------------------------
    # Plot only selected suppliers
    # --------------------------------------------------------

    selected_suppliers = []
    selected_costs = []

    if steel_amount_east_metal.value() > 0:
        selected_suppliers.append("East Metal Co.")
        selected_costs.append((steel_amount_east_metal.value() / 1000) * steel_price_east_metal)

    if steel_amount_sakura.value() > 0:
        selected_suppliers.append("Sakura Steelworks")
        selected_costs.append((steel_amount_sakura.value() / 1000) * steel_price_sakura)

    if steel_amount_black_forest.value() > 0:
        selected_suppliers.append("Black Forest Steel Co.")
        selected_costs.append((steel_amount_black_forest.value() / 1000) * steel_price_black_forest)

    if aluminium_amount_rising_sun.value() > 0:
        selected_suppliers.append("Rising Sun Aluminium")
        selected_costs.append((aluminium_amount_rising_sun.value() / 1000) * aluminium_price_rising_sun)

    if aluminium_amount_rhine_metal.value() > 0:
        selected_suppliers.append("RhineMetal Aluminium")
        selected_costs.append((aluminium_amount_rhine_metal.value() / 1000) * aluminium_price_rhine_metal)

    if aluminium_amount_fjordlight.value() > 0:
        selected_suppliers.append("Fjordlight Aluminium")
        selected_costs.append((aluminium_amount_fjordlight.value() / 1000) * aluminium_price_fjordlight)

    if microchips_amount_skytech.value() > 0:
        selected_suppliers.append("Skytech Microchips")
        selected_costs.append((microchips_amount_skytech.value() / 1500) * microchips_price_skytech)

    if microchips_amount_sic.value() > 0:
        selected_suppliers.append("Silicon Innovations")
        selected_costs.append((microchips_amount_sic.value() / 1500) * microchips_price_sic)

    if microchips_amount_nexgen.value() > 0:
        selected_suppliers.append("NexGen Microsystems")
        selected_costs.append((microchips_amount_nexgen.value() / 1500) * microchips_price_nexgen)

    if cobalt_amount_congo_cobalt.value() > 0:
        selected_suppliers.append("Congo Cobalt")
        selected_costs.append((cobalt_amount_congo_cobalt.value() / 1000) * cobalt_price_congo_cobalt)

    if cobalt_amount_auric_cobalt.value() > 0:
        selected_suppliers.append("AuricCobalt")
        selected_costs.append((cobalt_amount_auric_cobalt.value() / 1000) * cobalt_price_auric_cobalt)

    if cobalt_amount_ncc.value() > 0:
        selected_suppliers.append("NCC")
        selected_costs.append((cobalt_amount_ncc.value() / 1000) * cobalt_price_ncc)

    if lithium_amount_sollith.value() > 0:
        selected_suppliers.append("SolLith")
        selected_costs.append((lithium_amount_sollith.value() / 1000) * lithium_price_sollith)

    if lithium_amount_litio_andes.value() > 0:
        selected_suppliers.append("LitioAndes")
        selected_costs.append((lithium_amount_litio_andes.value() / 1000) * lithium_price_litio_andes)

    if lithium_amount_lithiumoz.value() > 0:
        selected_suppliers.append("LithiumOz")
        selected_costs.append((lithium_amount_lithiumoz.value() / 1000) * lithium_price_lithiumoz)

    plt.figure(figsize=(10, 5))
    plt.bar(selected_suppliers, selected_costs)

    plt.title("Selected Supplier Costs")
    plt.xlabel("Selected Suppliers")
    plt.ylabel("Cost")

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calculate_supplier_selection()
