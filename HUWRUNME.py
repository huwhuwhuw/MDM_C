import path_generation
import Indivual_Line_Score


# modify filepath
filepath = 'C:/Users/pigwi/Coding/MDM3/Transport/Bris_Codes_with_Weights_and_Coords_NEW.json'
with open(filepath, 'r') as file:
    import json
    Dict = json.load(file)

markov = [['E00073214', 'E00073211', 'E00074303', 'E00073500', 'E00074081', 'E00073510', 'E00073526', 'E00073446', 'E00073425', 'E00073431', 'E00174050', 'E00174284', 'E00174310', 'E00073436', 'E00174297', 'E00174231'],
          ['E00073214', 'E00073211', 'E00074303', 'E00073500', 'E00074081', 'E00073510', 'E00073526', 'E00073446', 'E00073425',
              'E00073431', 'E00073491', 'E00174050', 'E00174284', 'E00174310', 'E00073436', 'E00174297', 'E00174231'],
          ['E00075523', 'E00075536', 'E00075339', 'E00173041', 'E00173089', 'E00075692', 'E00075774', 'E00174323', 'E00073699', 'E00073856', 'E00173063', 'E00073390',
              'E00074192', 'E00073587', 'E00073584', 'E00074023', 'E00074025', 'E00174225', 'E00174316', 'E00174262', 'E00174242', 'E00174297', 'E00174231']
          ]

kmed = [['E00075790', 'E00075293', 'E00075659', 'E00073868', 'E00074120', 'E00075388', 'E00075587', 'E00075633', 'E00075774', 'E00173036', 'E00073600', 'E00073562', 'E00074178', 'E00073389', 'E00073403', 'E00074002', 'E00174316'],
        ['E00075506', 'E00075460', 'E00075388', 'E00074120', 'E00073868', 'E00075659', 'E00075293', 'E00075774',
            'E00173036', 'E00073600', 'E00073562', 'E00074178', 'E00073389', 'E00073403', 'E00074002', 'E00174316'],
        ['E00073219', 'E00073194', 'E00073957', 'E00074329', 'E00073543', 'E00073526',
            'E00073417', 'E00174307', 'E00174245', 'E00073234', 'E00073436', 'E00174316']
        ]


handler = path_generation.Data_Management()
distanceM = [0, 0, 0]
distanceK = [0, 0, 0]

# markov route statistics
for k, mRoute in enumerate(markov):
    for i, station in enumerate(mRoute):
        try:
            distanceM[k] += handler.load_route(mRoute[i],
                                               mRoute[i+1])['distance']
        except:
            print('route not found')

print('top 3 markov routes:')
print(f'distance: {distanceM}')
print(f'stop count: {[len(route) for route in markov]}')
print(f'score: {[Indivual_Line_Score.Line_Score(route, Dict)
      for route in markov]}')


# kmed route statistics
for k, mRoute in enumerate(kmed):
    for i, station in enumerate(mRoute):
        try:
            distanceM[k] += handler.load_route(mRoute[i],
                                               mRoute[i+1])['distance']
        except:
            print('route not found')

print('top 3 kmed routes:')
print(f'distance: {distanceM}')
print(f'stop count: {[len(route) for route in kmed]}')
print(f'score: {[Indivual_Line_Score.Line_Score(route, Dict)
      for route in kmed]}')
