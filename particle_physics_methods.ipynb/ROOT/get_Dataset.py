import requests
import os
import numpy as np
import h5py
import ROOT
from array import array


def depth(d):
	'''
	For each event, it finds the deepest layer in which the shower has deposited some E.
	Args:
	-----
		d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
		   that represent the 2d cell grids and the corresponding
		   E depositons.
	Returns:
	--------

	'''
	maxdepth = 2 * (d['layer_2'][:].sum(axis=(1,2)) != 0) 
	maxdepth[maxdepth == 0] = 1 * (d['layer_1'][:][maxdepth == 0].sum(axis=(1,2)) != 0) 
	return maxdepth


def total_energy(d):
	'''
	Calculates the total energy for each event across all layers.
	Args:
	-----
		d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
		   that represent the 2d cell grids and the corresponding
		   E depositons.
	'''
	return d['layer_0'][:].sum(axis=(1, 2)) + d['layer_1'][:].sum(axis=(1, 2)) + d['layer_2'][:].sum(axis=(1, 2))

def energy(layer, d):
	'''
	Finds total E deposited in a given layer for each event.
	Args:
	-----
		layer: int in {0, 1, 2} that labels the layer
		d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
		   that represent the 2d cell grids and the corresponding
		   E depositons.
    Returns:
    --------
	the total enery deposited in a given layer for each event
	'''
	return d['layer_{}'.format(layer)][:].sum(axis=(1, 2))


def efrac(elayer, total_energy):
	'''
	Finds the fraction of E deposited in a given layer for each event.
	Args:
	-----
		elayer: float, energy in a given layer for each event
		total_energy: float, total energy per event
	Returns:
	--------
		E_layer / E_total
	'''
	return	elayer / total_energy


def lateral_depth(d):
	'''
	Sum_{i} E_i * d_i
	'''
	return (d['layer_2'][:] * 2).sum(axis=(1,2)) + (d['layer_1'][:]).sum(axis=(1,2))


def lateral_depth2(d):
	'''
	Sum_{i} E_i * d_i^2
	'''
	return (d['layer_2'][:] * 2 * 2).sum(axis=(1,2)) + (d['layer_1'][:]).sum(axis=(1,2))


def shower_depth(lateral_depth, total_energy):
	'''
	lateral_depth / total_energy
	Args:
	-----
		lateral_depth: float, Sum_{i} E_i * d_i
		total_energy: float, total energy per event
	'''
	return lateral_depth / total_energy


def shower_depth_width(lateral_depth, lateral_depth2, total_energy):
	'''
	sqrt[lateral_depth2 / total_energy - (lateral_depth / total_energy)^2]
	Args:
	-----
		lateral_depth: float, Sum_{i} E_i * d_i
		lateral_depth2: float, Sum_{i} E_i * d_i * d_i
		total_energy: float, total energy per event
	'''
	return np.sqrt((lateral_depth2 / total_energy) - (lateral_depth / total_energy)**2)


def layer_lateral_width(layer, d):
	'''
	Args:
	-----
		layer: int in {0, 1, 2} that labels the layer
		d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
		   that represent the 2d cell grids and the corresponding
		   E depositons.
	'''
	e = energy(layer, d)
	eta_cells = {'layer_0' : 3, 'layer_1' : 12, 'layer_2' : 12}
	eta_bins = np.linspace(-240, 240, eta_cells['layer_' + str(layer)] + 1)
	bin_centers = (eta_bins[1:] + eta_bins[:-1]) / 2.
	x = (d['layer_{}'.format(layer)] * bin_centers.reshape(-1, 1)).sum(axis=(1,2))
	x2 = (d['layer_{}'.format(layer)] * (bin_centers.reshape(-1, 1) ** 2)).sum(axis=(1,2))

	N = len(e)
	lateralWidth0 = []
	for i in range(N):
		if e[i] > 0: 
			y = (x2[i] / e[i]) - ((x[i] / e[i]) ** 2)
			if y > 0:
				lateralWidth0.append(np.sqrt(y))
			else:
				lateralWidth0.append(-1)
		else:
			lateralWidth0.append(-1)
	
	return lateralWidth0




def hd5_to_root(data_filename, tree_name, root_file):
    data_file = h5py.File(data_filename, 'r')

    # ignore the "real" images, just get the variables
    """
    data_0 = data_file['layer_0'][:]
    data_1 = data_file['layer_1'][:]
    data_2 = data_file['layer_2'][:]

    # real_images = [data_0, data_1, data_2]
    sizes = [
        real_images[0].shape[1], real_images[0].shape[2],
        real_images[1].shape[1], real_images[1].shape[2],
        real_images[2].shape[1], real_images[2].shape[2]]
    """

    tree = ROOT.TTree(tree_name, tree_name)
    D = array('f', [0])
    E = array('f', [0])
    E0 = array('f', [0])
    E1 = array('f', [0])
    E2 = array('f', [0])

    E0frac = array('f', [0])
    E1frac = array('f', [0])
    E2frac = array('f', [0])

    lateralDepth = array('f', [0])
    lateralDepth2 = array('f', [0])

    showerDepth = array('f', [0])
    showerDepthWidth = array('f', [0])

    lateralWidth0 = array('f', [0])
    lateralWidth1 = array('f', [0])
    lateralWidth2 = array('f', [0])

    tree.Branch("depth", D, "depth")
    tree.Branch("E", E, "E")
    tree.Branch("E0", E0, "E0")
    tree.Branch("E1", E1, "E1")
    tree.Branch("E2", E2, "E2")

    tree.Branch("E0frac", E0frac, "E0frac")
    tree.Branch("E1frac", E1frac, "E1frac")
    tree.Branch("E2frac", E2frac, "E2frac")

    tree.Branch("lateralDepth", lateralDepth, "lateralDepth")
    tree.Branch("lateralDepth2", lateralDepth2, "lateralDepth2")

    tree.Branch("showerDepth", showerDepth, "showerDepth")
    tree.Branch("showerDepthWidth", showerDepthWidth, "showerDepthWidth")

    tree.Branch("lateralWidth0", lateralWidth0, "lateralWidth0")
    tree.Branch("lateralWidth1", lateralWidth1, "lateralWidth1")
    tree.Branch("lateralWidth2", lateralWidth2, "lateralWidth2")

    DepthArray = depth(data_file)
    EnergyArray = total_energy(data_file)
    E0Array = energy(0, data_file)
    E1Array = energy(1, data_file)
    E2Array = energy(2, data_file)

    lateralDepthArray = lateral_depth(data_file)
    lateralDepth2Array = lateral_depth2(data_file)

    lateralWidthArray0 = layer_lateral_width(0, data_file)
    lateralWidthArray1 = layer_lateral_width(1, data_file)
    lateralWidthArray2 = layer_lateral_width(2, data_file)

    for i in range(len(DepthArray)):
        D[0] = float(DepthArray[i])
        E[0] = EnergyArray[i]
        E0[0] = E0Array[i]
        E1[0] = E1Array[i]
        E2[0] = E2Array[i]

        E0frac[0] = E0[0] / E[0]
        E1frac[0] = E1[0] / E[0]
        E2frac[0] = E2[0] / E[0]

        lateralDepth[0] = lateralDepthArray[i]
        lateralDepth2[0] = lateralDepth2Array[i]

        showerDepth[0] = lateralDepth[0] / E[0]

        showerDepthWidth[0] = shower_depth_width(lateralDepth[0], lateralDepth2[0], E[0])

        lateralWidth0[0] = lateralWidthArray0[i]
        lateralWidth1[0] = lateralWidthArray1[i]
        lateralWidth2[0] = lateralWidthArray2[i]
        tree.Fill()

    tree.Write()

    
#-------------------
# main program
#------------------
    
directory_contents = os.listdir()
root_filename = 'calorimetry.root'

if root_filename not in directory_contents:
    f = ROOT.TFile(root_filename, 'RECREATE')

    ## electrons
    eplus_url = 'https://data.mendeley.com/public-files/datasets/pvn3xc3wy5/files/82030895-c680-432d-ac68-a6f2c8ed2641/file_downloaded'
    print('Fetching eplus.hdf5 from online...')
    eplus = requests.get(eplus_url)

    myFile = open('eplus.hdf5', 'wb')
    myFile.write(eplus.content)
    myFile.close()

    print('Converting to a ROOT file and removing the hdf5 file...')
    hd5_to_root('eplus.hdf5', 'eplus', f)
    os.remove('eplus.hdf5')

    ## photons
    gamma_url = 'https://data.mendeley.com/public-files/datasets/pvn3xc3wy5/files/1141aa57-edc0-477a-a32c-a38e5934b453/file_downloaded'
    print('Fetching gamma.hdf5 from online...')
    gamma = requests.get(gamma_url)

    myFile = open('gamma.hdf5', 'wb')
    myFile.write(gamma.content)
    myFile.close()
    
    print('Converting to a ROOT file and removing the hdf5 file...')
    hd5_to_root('gamma.hdf5', 'gamma', f)
    os.remove('gamma.hdf5')

    ## pions
    piplus_url = 'https://data.mendeley.com/public-files/datasets/pvn3xc3wy5/files/292fa8da-1877-482b-9fd3-df237d43b142/file_downloaded'
    print('Fetching piplus.hdf5 from online...')
    piplus = requests.get(piplus_url)

    myFile = open('piplus.hdf5', 'wb')
    myFile.write(piplus.content)
    myFile.close()
    
    print('Converting to a ROOT file and removing the hdf5 file...')
    hd5_to_root('piplus.hdf5', 'piplus', f)
    os.remove('piplus.hdf5')
    
    # Fecha o arquivo ROOT para garantir que todos os dados sejam salvos
    f.Close()

    print(f"Arquivo '{root_filename}' salvo com sucesso.")