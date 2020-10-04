import argparse

synonymous_acronyms = [['lamina papyracea', 'lamina paprycea'],
    ['permissive hypercapnia', 'permissive hypercapnea'],
    ['spider angiomas', 'spider angiomata', 'spider angioma'],
    ['IGNORE', 'beam hardening artifacts', 'beam hardening artifact'], ###
    ['IGNORE', 'human immunodefiency virus', 'human immunodeficiency virus'], ###
    ['caput medusa', 'caput medusae'],
    ['fiducial seed', 'fiducial seeds'],
    ['corpus callosum', 'corpus collosum'],
    ['IGNORE', 'weekly transdermal qwed', 'weekly transdermal qfri', 'weekly transdermal qmon', 'weekly transdermal qsat'], ###
    ['paracolic gutters', 'pericolic gutters', 'paracolic gutter'],
    ['IGNORE', 'perc plcmt gastrosotmy', 'perc plcmt gastromy'], ###
    ['creatine kinases', 'creatine kinase'],
    ['pulsus paradoxus', 'pulsus parodoxus'],
    ['IGNORE', 'operating heavy machinery', 'operate heavy machinery'], ###
    ['bullous pemph', 'bullous pemphigoid'],
    ['enhances homogeneously', 'enhances homogenously'],
    ['IGNORE', 'exuberant callus formation'], ###
    ['IGNORE', 'patent ductus arteriosus', 'patent ductus arteriosis'], ###
    ['abo incompatible', 'abo incompatibility'],
    ['IGNORE', 'dietary indiscretions', 'dietary indiscretion', 'dietary indescretion'], ###
    ['IGNORE', 'prolene mesh overlay'], ###
    ['IGNORE', 'suspicious clustered microcalcifications', 'suspicious clustered microcalcification'], ###
    ['IGNORE', 'plantar calcaneal spurs', 'plantar calcaneal spur'], ###
    ['talc pleurodesis', 'talc pleuradesis'],
    ['tensilon pearls', 'tessalon perles'],
    ['IGNORE', 'pneumocystis jirovecii', 'pneumocystis jiroveci', 'pneumocystis jirvovecii'], ###
    ['IGNORE', 'drain excluding append', 'drain excluding appendiceal', 'drain excluding appendicealclip'], ###
    ['percutaneous transhepatic cholangiography', 'perc transhepatic cholangiogra', 'percutaneous transhepatic cholangio'],
    ['IGNORE', 'lentiform nucleus', 'lentiform nuclei'], ###
    ['IGNORE', 'dichorionic diamniotic twin', 'dichorionic diamniotic twins', 'diamniotic dichorionic twin'], ###
    ['interlobular septae', 'interlobular septa'],
    ['IGNORE', 'intro perc trnashepatic', 'intro perc tranhepatic', 'intro perc tranhep'], ###
    ['lactated ringer', 'lactated ringers'],
    ['IGNORE', 'torsade de pointes', 'torsades de pointes'], ###
    ['IGNORE', 'digital screening mammography', 'digital screening mammogram'], ###
    ['IGNORE', 'choroid plexus cysts', 'choroid plexus cyst'], ###
    ['IGNORE', 'ischial tuberosities', 'ischial tuberosity'], ###
    ['IGNORE', 'diamniotic dichorionic', 'dichorionic diamniotic'], ###
    ['IGNORE', 'palatine tonsils', 'palatine tonsil'], ###
    ['IGNORE', 'motor vehicke accident', 'motor vehicle accidemt'], ###
    ['IGNORE', 'hospital cashier recipt'], ###
    ['IGNORE', 'homonymous hemianopia', 'homonymous hemianopsia'], ###
    ['IGNORE', 'thrombotic thrombocytopenic purpura', 'thrombotic thrombocytopenic pupura'], ###
    ['IGNORE', 'video assisted thoracoscopic', 'video assisted thoracoscopy'], ###
    ['IGNORE', 'salpingo oophorectomy']] ###

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--results_file",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()

inp = open(args.results_file, 'r')
lines = inp.readlines()
inp.close()

i = 0
correct = 0
incorrect = 0
ignored = 0
total = 0

print(synonymous_acronyms)
while len(lines) >= (i+3):
    generated = lines[i].replace('Generated 0: ', '').strip()
    expected = lines[i+1].replace('Expected: ', '').strip()

    ignore = False
    synonym = False
    for syn_list in synonymous_acronyms:
        if expected in syn_list:
            if 'IGNORE' in syn_list:
                ignore = True
                break
            elif generated in syn_list:
                synonym = True
                break


    i += 5

    if ignore:
        ignored += 1
        continue

    total += 1
    if synonym:
        correct += 1
    elif generated == expected:
        correct += 1
    else:
        incorrect += 1


print('Number correct : ' + str(correct))
print('Number incorrect : ' + str(incorrect))
print('Total predictions : ' + str(total))
print('Accuracy: ' + str(correct/total))
print('Number ignored : ' + str(ignored))
