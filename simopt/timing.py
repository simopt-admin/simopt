import cProfile
import pstats
import io
#import run_experiments


pr = cProfile.Profile()
pr.enable()

#exec(open("run_experiments.py").read())
exec(open("timing_bootstrap.py").read())

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('profile_results.txt', 'w+') as f:
    f.write(s.getvalue())