/*
 *Very Large Integer Library, License - Version 1.0 - May 3rd, 2012
 *
 *Timothee Ewart - University of Geneva,
 *Andreas Hehn - Swiss Federal Institute of technology Zurich.
 *Maxim Milakov - NVIDIA
 *
 *Permission is hereby granted, free of charge, to any person or organization
 *obtaining a copy of the software and accompanying documentation covered by
 *this license (the "Software") to use, reproduce, display, distribute,
 *execute, and transmit the Software, and to prepare derivative works of the
 *Software, and to permit third-parties to whom the Software is furnished to
 *do so, all subject to the following:
 *
 *The copyright notices in the Software and this entire statement, including
 *the above license grant, this restriction and the following disclaimer,
 *must be included in all copies of the Software, in whole or in part, and
 *all derivative works of the Software, unless such copies or derivative
 *works are solely in the form of machine-executable object code generated by
 *a source language processor.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 *SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 *FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 *ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *DEALINGS IN THE SOFTWARE.
 */

namespace vli {
    namespace detail {

    template <std::size_t NumBits, int Order, int NumVars>
    tasklist_keep_order<NumBits, max_order_each<Order>, NumVars>::tasklist_keep_order(){
        // As templated this array will be allocated a couple of time for every tupple of the cmake global size negligible  
        // only once due to singleton
        gpu::cu_check_error(cudaMalloc((void**)&(this->execution_plan_), mul_block_size<max_order_each<2*Order>, NumVars>::value*max_iteration_count<max_order_each<2*Order>, NumVars>::value*sizeof(single_coefficient_task)),__LINE__);
        gpu::cu_check_error(cudaMalloc((void**)&(this->workblock_count_by_warp_), mul_block_size<max_order_each<2*Order>, NumVars>::value/32*sizeof(int)),__LINE__);
        element_count_prepared=0;
        plan();
    }

    template <std::size_t NumBits, int Order, int NumVars>
    void tasklist_keep_order<NumBits, max_order_each<Order>, NumVars>::plan(){
        std::vector<int> workblock_count_by_warp_local(mul_block_size<max_order_each<2*Order>, NumVars>::value / 32U,0);
        std::vector<int> work_total_by_size(mul_block_size<max_order_each<2*Order>, NumVars>::value / 32U,0);
        // TO DO CHECK stride and result_stride, there is pb with the Var argument
        std::vector<vli::detail::single_coefficient_task > tasks(((result_stride<0,NumVars,Order>::value*result_stride<1,NumVars,Order>::value*result_stride<2,NumVars,Order>::value*result_stride<3,NumVars,Order>::value + 32U - 1) / 32U) * 32U);
        for(unsigned int degree_w = 0; degree_w <result_stride<3,NumVars, Order>::value; ++degree_w) {
            for(unsigned int degree_z = 0; degree_z <result_stride<2,NumVars, Order>::value; ++degree_z) {
                for(unsigned int degree_y = 0; degree_y <result_stride<1,NumVars, Order>::value; ++degree_y) {
                    for(unsigned int degree_x = 0; degree_x <result_stride<0,NumVars, Order>::value; ++degree_x) {
                        vli::detail::single_coefficient_task& task = tasks[  degree_w * result_stride<1,NumVars, Order>::value * result_stride<2,NumVars, Order>::value * result_stride<3,NumVars, Order>::value
                                                                           + degree_z * result_stride<1,NumVars, Order>::value * result_stride<2,NumVars, Order>::value 
                                                                           + degree_y * result_stride<1,NumVars, Order>::value
                                                                           + degree_x];
                        task.output_degree_x = degree_x;
                        task.output_degree_y = degree_y;
                        task.output_degree_z = degree_z;
                        task.output_degree_w = degree_w;
                        task.step_count =   (std::min<int>((result_stride<0, NumVars ,Order>::value - 1) - degree_x, degree_x) + 1)
                                          * (std::min<int>((result_stride<1, NumVars ,Order>::value - 1) - degree_y, degree_y) + 1) 
                                          * (std::min<int>((result_stride<2, NumVars ,Order>::value - 1) - degree_z, degree_z) + 1) 
                                          * (std::min<int>((result_stride<3, NumVars ,Order>::value - 1) - degree_w, degree_w) + 1);
                    }
                }
            }
        }
        // Fill the task list up to the multiple of the warp size
        for(unsigned int i = result_stride<0, NumVars, Order>::value*result_stride<1, NumVars, Order>::value*result_stride<2, NumVars, Order>::value*result_stride<3, NumVars, Order>::value; i < tasks.size(); ++i) {
               vli::detail::single_coefficient_task& task = tasks[i];
               task.output_degree_x = 0;
               task.output_degree_y = 0;
               task.output_degree_z = 0;
               task.output_degree_w = 0;
               task.step_count = 0;
        }
        // Sort the tasks in step_count descending order
        std::sort(tasks.begin(), tasks.end(), vli::detail::single_coefficient_task_sort);
        std::vector<vli::detail::single_coefficient_task > tasks_reordered(mul_block_size<max_order_each<2*Order>, NumVars>::value * max_iteration_count<max_order_each<2*Order>, NumVars>::value);
        // this thing should be generic ... yes it is ! 
        for(unsigned int batch_id = 0; batch_id < tasks.size() / 32; ++batch_id) {
               int warp_id = std::min_element(work_total_by_size.begin(), work_total_by_size.end()) - work_total_by_size.begin(); // - to get the position
                std::copy(
               	tasks.begin() + (batch_id * 32),
               	tasks.begin() + ((batch_id + 1) * 32),
               	tasks_reordered.begin() + (workblock_count_by_warp_local[warp_id] * mul_block_size<max_order_each<2*Order>, NumVars>::value) + (warp_id * 32));
        
               int max_step_count = tasks[batch_id * 32].step_count;
               workblock_count_by_warp_local[warp_id]++;
               work_total_by_size[warp_id] += max_step_count;
        }
	gpu::cu_check_error(cudaMemcpyAsync(workblock_count_by_warp_, &(*workblock_count_by_warp_local.begin()), sizeof(int) * workblock_count_by_warp_local.size(), cudaMemcpyHostToDevice),__LINE__);
	gpu::cu_check_error(cudaMemcpyAsync(execution_plan_, &(*tasks_reordered.begin()), sizeof(single_coefficient_task) * tasks_reordered.size(),cudaMemcpyHostToDevice),__LINE__);
    }

    } // end namespace detail
 }//end namespace vli
