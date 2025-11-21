import importlib, sys
pkgs = ['streamlit', 'folium', 'streamlit_folium', 'PIL']
for p in pkgs:
    try:
        importlib.import_module(p if p != 'PIL' else 'PIL.Image')
        print(p + ' OK')
    except Exception as e:
        print(p + ' ERROR: ' + str(e))
        sys.exit(2)
print('All imports OK')
