### How to compile documentation:

Execute the following in order to compile documentation: 

	make html
    
### How to open documentation:
Open **build/html/yateto.html** file with your favorite web-browser.

For example:

	firefox ./build/html/yateto.html

### How to get documentation coverage statistics:
Generate statistics:

    sphinx-build -b coverage source/ build/coverage
    
Read the following file to get a list of all undocumentated classes and methods:

    cat ./build/coverage/python.txt

Execute the following python script to get numerical statistics:

    python ./undoc_stats.py


### How to fully update documentation (only for documentation developers):
From time to time one has to generate new *rst template files* based on the source files of the project. 
In order to achive this, type the following:

	SPHINX_APIDOC_OPTIONS=members sphinx-apidoc -f -o ./source ..
where
- SPHINX_APIDOC_OPTIONS=members - forces to switch off *undoc-members* tags which obscure documentation code coverage statistics
- sphinx-apidoc - rst templates generator
- -f - forces to rewrite *rst template files* (the default implementation prevents it)
- -o ./source - means that the output directory is ./source
- .. - specifies the location of the input (source code) directory
