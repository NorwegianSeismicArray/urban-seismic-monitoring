<a name="readme-top"></a>

# Urban seismic monitoring

Code and examples related to the submitted paper **Monitoring urban construction and quarry blasts with low-cost seismic sensors and machine learning tools in the city of Oslo, Norway**.

This code will be updated and more documentation will be added. Since wavefrom data is not yet available via FDSN server / EIDA node, currently the code needs to be adapted for new data the user may have. Soon the data used in the paper will be openly available and then the code will run as it is.

## Outlier detection with auto-encoder

To install requirements :
```
conda create -n urbanmon python=3.10 obspy cartopy pytest pytest-json
conda activate urbanmon
pip install tensorflow
pip install tslearn
pip install tqdm
pip install pandas
pip install keras-tuner
```

Incase of library trouble :

ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path-to-your-env/urbanmon/lib/
```

To train auto-encoder for outlier detection (edit scripts for different stations):
```
python create_training_data_AutoEncoder.py
python train_AutoEncoder.py 
```

To run outlier detection (whole time period or two days):
```
python run_outlier_detection.py OSLN2 2022-06-01T00:00:00 2023-09-28T00:00:00 4.0 0.78
python run_outlier_detection.py OSLN2 2022-10-10T00:00:00 2022-10-12T00:00:00 4.0 0.78
```

To locate and plot outlier events:
```
python locate_events.py OSLN2 2022-10-10 outlier
```

To locate and plot sta/lta events:
```
python locate_events.py OSLN2 2022-10-10 stalta
```

## Supervised blast classification


To create training data for blast classification:
```
python create_training_data_BlastClassifier.py
```

To train blast classifier:
```
python train_BlastClassifier.py
```

To run the blast classifier:
```
python run_blast_classifier.py
```

To locate and plot blast classifications:
```
python locate_events.py OSLN2 2022-10-10 classifier
```




## License

See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Andreas Köhler - andreas.kohler@norsar.no - [ORCID](https://orcid.org/0000-0002-1060-7637)

Erik B. Myklebust - [ORCID](https://orcid.org/0000-0002-3056-2544)


Project Link: [https://github.com/NorwegianSeismicArray/urban-seismic-monitoring](https://github.com/NorwegianSeismicArray/urban-seismic-monitoring)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Models are built with [TensorFLow](https://www.tensorflow.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

