import subprocess

path = '/data-sinno/c.siargkas/datasets/greek/tedx/data/greek_test_clips/000881.wav'
subprocess.run(["ffplay", "-loglevel", "9", path])