import os
import numpy as np
import pymesh


class Protein:

    def __init__(self, id, pdb_filename, vertices, faces, normals, vertices_features, patch_radius, patches_features,
                 rhos, thetas,
                 neigh_indices, mask, iface):

        self.id = id
        self.pdb_filename = pdb_filename
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        self.vertices_features = vertices_features
        self.patch_radius = patch_radius
        self.patches_features = patches_features
        self.rhos = rhos
        self.thetas = thetas
        self.neigh_indices = neigh_indices
        self.mask = mask
        self.fingerprints = []
        self.patch2amino = {}
        self.amino2seqpos = {}
        self.full_sequence = ""
        self.iface = {"kdtree": iface}
        # self.get_patches_residues()

    def normalize_electrostatics(self, lower_th=-30, upper_th=+30):
        elec = self.patches_features[:, :, 3]
        elec[elec > upper_th] = upper_th
        elec[elec < lower_th] = lower_th
        elec = elec - lower_th
        elec = elec / (upper_th - lower_th)
        self.patches_features[:, :, 3] = 2 * elec - 1

        elec = self.vertices_features[:, 2]
        elec[elec > upper_th] = upper_th
        elec[elec < lower_th] = lower_th
        elec = elec - lower_th
        elec = elec / (upper_th - lower_th)
        self.vertices_features[:, 2] = 2 * elec - 1

    def normalize_hydrophobia(self):
        self.vertices_features[:, 1] /= 4.5
        self.patches_features[:, :, 2] /= 4.5

    def save_ply_masif(
            self,
            patches_idxs=None,
            iface=None,
            iface_type="kdtree",
            name_extension="",
    ):
        """ Save vertices, mesh in ply format.
            vertices: coordinates of vertices
            faces: mesh
        """
        surfaces_path = "/media/raoufks/Maxtor/raouf/data/surfaces"

        if not os.path.exists(surfaces_path):
            os.makedirs(surfaces_path)

        vertices, faces, normals = None, None, None
        shape_index, ddc, hphob, electrostatics, hbond = None, None, None, None, None
        rho, theta = None, None
        filename = None

        iface = iface.reshape(-1) if iface is not None else \
            self.iface[iface_type].reshape(-1) if self.iface[iface_type] is not None \
                else None

        if patches_idxs is not None:
            vertices_idx = []
            for patch_idx in patches_idxs:
                # vertices
                vertices_idx.append(self.neigh_indices[patch_idx])
            vertices_idx = np.concatenate(vertices_idx)
            vertices_idx = list(set(vertices_idx))
            vertices = self.vertices[vertices_idx]

            # normals
            normals = self.normals[vertices_idx]

            # faces
            ## to check in a complexity O(1) through hashing
            s = set(vertices_idx)
            ## mapping vertices
            map_prot_to_patch = {vertices_idx[i]: i for i in range(len(vertices_idx))}
            ## detect corresponding faces
            faces = np.array([[map_prot_to_patch[s1],
                               map_prot_to_patch[s2],
                               map_prot_to_patch[s3]] for s1, s2, s3 in self.faces
                              if s1 in s and s2 in s and s3 in s])

            # features
            shape_index = self.vertices_features[vertices_idx, 0]
            # ddc = self.patches_features[patch_idx, self.mask[patch_idx].astype(bool), 1]
            hphob = self.vertices_features[vertices_idx, 1]
            electrostatics = self.vertices_features[vertices_idx, 2]
            hbond = self.vertices_features[vertices_idx, 3]

            if iface is not None:
                iface = iface[vertices_idx]
            str_patches_idx = list(map(lambda x: str(x), patches_idxs))
            str_filename = "+".join(str_patches_idx)
            str_filename = "meta" if len(str_filename) > 30 else str_filename
            filename = os.path.join(surfaces_path, self.id + "patch_" + str_filename + name_extension + ".ply")


        else:
            vertices = self.vertices
            faces = self.faces
            normals = self.normals

            shape_index = self.vertices_features[:, 0]
            hphob = self.vertices_features[:, 1]
            electrostatics = self.vertices_features[:, 2]
            hbond = self.vertices_features[:, 3]
            filename = os.path.join(surfaces_path, self.id + name_extension + ".ply")

        print("saving : ", filename)

        mesh = pymesh.form_mesh(vertices, faces)
        if normals is not None:
            n1 = normals[:, 0]
            n2 = normals[:, 1]
            n3 = normals[:, 2]
            mesh.add_attribute("vertex_nx")
            mesh.set_attribute("vertex_nx", n1)
            mesh.add_attribute("vertex_ny")
            mesh.set_attribute("vertex_ny", n2)
            mesh.add_attribute("vertex_nz")
            mesh.set_attribute("vertex_nz", n3)
        # print("normals ",normals.shape)

        if shape_index is not None:
            mesh.add_attribute("vertex_si")
            mesh.set_attribute("vertex_si", shape_index)
        # print("shape_index ", shape_index.shape)

        if ddc is not None:
            mesh.add_attribute("vertex_ddc")
            mesh.set_attribute("vertex_ddc", ddc)
        # print("shape_index ", ddc.shape)

        if electrostatics is not None:
            mesh.add_attribute("charge")
            mesh.set_attribute("charge", electrostatics)
        # print("electrostatics ", electrostatics.shape)

        if hbond is not None:
            mesh.add_attribute("hbond")
            mesh.set_attribute("hbond", hbond)
        # print("hbond ", hbond.shape)

        if hphob is not None:
            mesh.add_attribute("vertex_hphob")
            mesh.set_attribute("vertex_hphob", hphob)
        # print("hphob ", hphob.shape)

        if rho is not None:
            mesh.add_attribute("rho")
            mesh.set_attribute("rho", rho)

        if theta is not None:
            mesh.add_attribute("theta")
            mesh.set_attribute("theta", theta)

        if iface is not None:
            mesh.add_attribute("vertex_iface")
            mesh.set_attribute("vertex_iface", iface)

        pymesh.save_mesh(filename, mesh, *mesh.get_attribute_names(), ascii=True, use_float=True)
