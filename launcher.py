# launcher.py
import os
from streamlit.web import bootstrap  # stable entrypoint
APP = os.path.join(os.path.dirname(__file__), "app.py")
bootstrap.run(APP, "", [], {})  # script_path, command_line, args, flag_options
