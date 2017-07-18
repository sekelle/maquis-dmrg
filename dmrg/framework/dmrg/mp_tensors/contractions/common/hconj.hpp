            std::size_t save = 0;
            for (index_type b=0; b<ret.aux_dim(); ++b)
            {
                if (mpo.herm_info.right_conj(b) == b)
                {
                    for(index_type k=0; k<ret[b].n_blocks(); ++k)
                    {
                        charge lc = ret[b].basis().left_charge(k);
                        charge rc = ret[b].basis().right_charge(k); 
                        if (lc != rc)
                        {
                            save += ret[b].basis().left_size(k) * ret[b].basis().right_size(k);
                            int S = mpo.right_spin(b).get();
                            typename Matrix::
                            value_type scale = ::SU2::conjugate_correction<typename Matrix::value_type, SymmGroup>
                                                (lc, rc, S);

                            index_type k2 = ret[b].find_block(rc, lc);
                            OtherMatrix B = transpose(ret[b][k2] * scale);
                            if (norm_square(ret[b][k] - B) > 1e-6)
                            {
                                maquis::cout << lc << rc << std::endl;
                                maquis::cout << ret[b][k] << std::endl;
                                maquis::cout << B << std::endl;
                                exit(1);
                            }
                        }
                    }
                }
            }

            maquis::cout << "save " << 8*save << " " << size_of(ret) << " " << double(8*save) / size_of(ret) << std::endl;
            return ret;

