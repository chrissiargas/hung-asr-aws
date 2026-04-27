import subprocess

path = '/data-sinno/c.siargkas/datasets/greek/tedx/data/greek_test_clips/000888.wav'
subprocess.run(["ffplay", "-loglevel", "0", path])