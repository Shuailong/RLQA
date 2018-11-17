import pstats
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'train_profile.txt'
p = pstats.Stats(filename)
# p.sort_stats('cumulative').print_stats(10)
p.strip_dirs().sort_stats('tottime', 'ncalls').print_stats(30)
