${PYTHON} setup.py --skip-klb-download build_ext -I${PREFIX}/include/klb -L${PREFIX}/lib

# The extra args here mean "Don't install a zipped .egg, just install the real files."
${PYTHON} setup.py --skip-klb-download install --single-version-externally-managed --record=/dev/null
