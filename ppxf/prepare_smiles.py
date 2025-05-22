import numpy as np
from astropy.io import fits
from pathlib import Path

ppxf_dir = Path('./')

IMF_types = ['Bimodal', 'Chabrier_1.3', 'Revised_Kroupa', 'Unimodal', 'Universal_Kroupa']

alpha_enhances = ['aFem02', 'aFep00', 'aFep02', 'aFep04', 'aFep06']

IMF = 'Chabrier_1.3'

smiles_glob = '**/sMILES_SSPs/{}/*/*'.format(IMF)
files = ppxf_dir.glob(smiles_glob)

ages = []
metallicities = []
alphas = []

templates = []

for file in files:
	age = float(file.name.split('_')[0].split('T')[1])
	metallicity = float(file.name.split('Z')[1].split('T')[0].replace('m','-').replace('p',''))
	alpha = float(file.parent.name.replace('aFe', '').replace('m','-').replace('p','')) / 10
	ages.append(age)
	metallicities.append(metallicity)
	alphas.append(alpha)
	data = fits.getdata(file)
	templates.append(data)
	
sorted_data = sorted(zip(ages, metallicities, alphas, templates), key=lambda x: (x[0], x[1], x[2]))
sorted_ages, sorted_metallicities, sorted_alphas, sorted_spectra = zip(*sorted_data)

templates_arranged = np.array(sorted_spectra).T
required_dimensions = [templates_arranged.shape[0],len(np.unique(sorted_ages)),len(np.unique(sorted_metallicities)),len(np.unique(sorted_alphas))]
templates_out = templates_arranged.reshape(required_dimensions)

ages_out = np.unique(sorted_ages)
metallicities_out = np.unique(sorted_metallicities)
alphas_out = np.unique(sorted_alphas)

fwhm = [2.5] * templates_out.shape[0]
lam = np.linspace(3540.5,7409.6,templates_out.shape[0])
masses = np.ones(templates_out.shape[1:])

np.savez_compressed(ppxf_dir / 'ppxf/sps_models/spectra_smiles_{}_9.0.npz'.format(IMF), templates=templates_out, masses=masses, 
                                lam=lam, ages=ages_out, metals=metallicities_out, alphas=alphas_out, fwhm=fwhm)

# ages_to_plot = [age in np.arange(2,16,2) for age in np.unique(sorted_ages)]
# metallicities_to_plot = np.unique(sorted_metallicities) == 0.06
# plt.plot(lam,templates_repackaged[:,ages_to_plot,metallicities_to_plot])
# plt.show()