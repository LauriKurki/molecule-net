
#include <cmath>
#include <cstdio>
#include <vector>

#define CUTOFF 4 // Cutoff in standard deviations for evaluating the Gaussian exponent

extern "C" {

    void peak_dist(
        int nb, int nx, int ny, int nz, float *dist,
        int *N_atom, float *pos,
        float xyz_start[3], float xyz_step[3], float std
    ) {

        int nxyz = nx * ny * nz;
        float cov_inv = 1 / (std * std);
        float cutoff2 = CUTOFF * CUTOFF * std * std;

        float pi = 2 * acos(0.0);
        float denom = sqrt(2*pi) * std;
        float prefactor = 1 / (denom * denom * denom);

        int ind = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    float grid_xyz[3] = {
                        xyz_start[0] + i * xyz_step[0],
                        xyz_start[1] + j * xyz_step[1],
                        xyz_start[2] + k * xyz_step[2]
                    };
                    int ip0 = 0;
                    for (int b = 0; b < nb; b++) {
                        float v = 0;
                        int ip = 0;
                        for (; ip < N_atom[b]; ip++) {
                            float atom_xyz[3] = {pos[3*(ip0 + ip)], pos[3*(ip0 + ip)+1], pos[3*(ip0 + ip)+2]};
                            float dp[3] = {
                                (grid_xyz[0] - atom_xyz[0]),
                                (grid_xyz[1] - atom_xyz[1]),
                                (grid_xyz[2] - atom_xyz[2])
                            };
                            float dp2 = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
                            if (dp2 < cutoff2) {
                                v += exp(-0.5 * dp2 * cov_inv);
                            }
                        }
                        v *= prefactor;
                        ip0 += ip;
                        dist[b * nxyz + ind] = v;
                    }
                    ind++;
                }
            }
        }

    }

    void peak_dist_species(
        int nb, int nx, int ny, int nz, 
        float *dist_species, // Output: size nb * 5 * nx * ny * nz
        int *N_atom, 
        float *pos, // Input: shape (N x 4)
        float xyz_start[3], float xyz_step[3], float std
    ) {
        const int species[] = {1, 6, 7, 8, 9}; // Atomic species
        const int num_species = 5;

        int nxyz = nx * ny * nz;
        float cov_inv = 1 / (std * std);
        float cutoff2 = CUTOFF * CUTOFF * std * std;

        float pi = 2 * acos(0.0);
        float denom = sqrt(2 * pi) * std;
        float prefactor = 1 / (denom * denom * denom);

        // Initialize output
        for (int b = 0; b < nb; b++) {
            for (int s = 0; s < num_species; s++) {
                for (int i = 0; i < nxyz; i++) {
                    dist_species[(b * num_species + s) * nxyz + i] = 0.0f;
                }
            }
        }

        int ind = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    float grid_xyz[3] = {
                        xyz_start[0] + i * xyz_step[0],
                        xyz_start[1] + j * xyz_step[1],
                        xyz_start[2] + k * xyz_step[2]
                    };
                    int ip0 = 0;
                    for (int b = 0; b < nb; b++) {
                        std::vector<float> v_species(num_species, 0.0f);
                        int ip = 0;
                        for (; ip < N_atom[b]; ip++) {
                            float atom_xyz[3] = {
                                pos[4 * (ip0 + ip)], 
                                pos[4 * (ip0 + ip) + 1], 
                                pos[4 * (ip0 + ip) + 2]
                            };
                            int atom_type = (int)pos[4 * (ip0 + ip) + 3];
                            
                            // Find species index
                            int species_index = -1;
                            for (int s = 0; s < num_species; s++) {
                                if (atom_type == species[s]) {
                                    species_index = s;
                                    break;
                                }
                            }
                            if (species_index == -1) continue; // Skip if species is unknown

                            // Compute distance squared
                            float dp[3] = {
                                (grid_xyz[0] - atom_xyz[0]),
                                (grid_xyz[1] - atom_xyz[1]),
                                (grid_xyz[2] - atom_xyz[2])
                            };
                            float dp2 = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
                            if (dp2 < cutoff2) {
                                v_species[species_index] += exp(-0.5 * dp2 * cov_inv);
                            }
                        }
                        ip0 += ip;

                        // Store results
                        for (int s = 0; s < num_species; s++) {
                            dist_species[(b * num_species + s) * nxyz + ind] = v_species[s] * prefactor;
                        }
                    }
                    ind++;
                }
            }
        }
    }
}