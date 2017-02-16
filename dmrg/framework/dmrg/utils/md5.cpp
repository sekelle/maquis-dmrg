/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2017 Stanford University Department of Chemistry
 *                    Sebastian Keller <sebkelle@phys.ethz.ch>
 * 
 * This software is part of the ALPS Applications, published under the ALPS
 * Application License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Application License along with
 * the ALPS Applications; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/


#include <openssl/md5.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>

std::string md5sum(std::string fname)
{
    MD5_CTX ctx;
    MD5_Init(&ctx);

    //std::ifstream ifs(fname, std::ios::binary);
    std::ifstream ifs(fname.c_str());
    
    char file_buffer[4096];
    while (ifs.read(file_buffer, sizeof(file_buffer)) || ifs.gcount()) {
        MD5_Update(&ctx, file_buffer, ifs.gcount());
    }
    unsigned char digest[MD5_DIGEST_LENGTH] = {};
    MD5_Final(digest, &ctx);
    
    std::ostringstream ss;
    for(unsigned i=0; i <MD5_DIGEST_LENGTH; i++) {
        ss << std::hex << static_cast<int>(digest[i]);
    }
    
    return ss.str();
}
