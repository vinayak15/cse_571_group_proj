import os
iteration = 1
array =  [3,5,10,15,25,50,100,120]
# for i in array:
#     os.system('python pacman.py -p TrueOnlineSarsaLamda -a extractor=ImprovedExtractor -x {0} -n {1} -l mediumClassic -q'.format(i, i+50))
#
# for i in array:
#     os.system('python pacman.py -p ApproximateQAgent -a extractor=ImprovedExtractor -x {0} -n {1} -l mediumClassic -q'.format(i, i+50))
#
# for i in array:
#     os.system('python pacman.py -p ApproximateSarsaAgent -a extractor=ImprovedExtractor -x {0} -n {1} -l mediumClassic -q'.format(i, i+50))


for i in array:
    os.system('python pacman.py -p TrueOnlineSarsaLamda -a extractor=ImprovedExtractor -x {0} -n {1} -l powerClassic -q'.format(i, i+25))

for i in array:
    os.system('python pacman.py -p ApproximateQAgent -a extractor=ImprovedExtractor -x {0} -n {1} -l powerClassic -q'.format(i, i+25))

for i in array:
    os.system('python pacman.py -p ApproximateSarsaAgent -a extractor=ImprovedExtractor -x {0} -n {1} -l powerClassic -q'.format(i, i+25))

# for i in range(iteration):
#     value = os.system('python pacman.py -p ApproximateSarsaAgent -a extractor=ImprovedExtractor -x 400 -n 450 -l mediumClassic -q')
# print('next iteration')
# for i in range(iteration):
#     value = os.system('python pacman.py -p ApproximateSarsaAgent -a extractor=ImprovedExtractor -x 400 -n 450 -l powerClassic -q')