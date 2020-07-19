 The directory contains requirements.txt file containing the required libraries.
 To run the program, follow these steps:
 1. Download NOTEEVENTS.csv and ADMISSIONS.csv and PATIENTS.csv files.
 2. The code can be checked out from the GitHub repository:
    git clone https://github.com/vaivaug/Dissertation.git
 3. The directory contains requirements.txt file containing the required libraries
        cd Dissertation
        pip install -r requirements.txt
 4. Place the downloaded files in the same directory as the Dissertation folder.
 5. a) Inside the Main.py file, set the value set_parameters_gui to be 'True' in
       order to set the program parameters from Gui
    b) Inside the Main.py file, set the value set_parameters_gui to be 'False' in
    order to set the program parameters inside the Main.py file.
 6. Some more parameters can be edited in the 'program_run_start_to_end' file. Those are:
    age filtering
    multiple diseases filtering
    perform prediction on validation or test set (when not doing cross validaiton)
 7. Run the program:
    python Main.py