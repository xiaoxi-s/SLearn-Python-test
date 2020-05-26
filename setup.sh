echo "test slearn on python end"

# function for updating slearn
# The reason for uninstalling slearn is because 
# the version could be the same during development.
# So the update option might not work
function update_slearn {
    # get current working directory
    current_directory=$(pwd)
    # go to the directory where the slearn source is
    cd /media/adrian/Envir/workspace/vscode/cpp/SLearn/

    # update
    sudo pip3 uninstall --yes slearn
    sudo python3 setup.py install

    cd $current_directory
}

update_slearn
