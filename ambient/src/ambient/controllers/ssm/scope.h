/*
 * Ambient, License - Version 1.0 - May 3rd, 2012
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef AMBIENT_CONTROLLERS_SSM_SCOPE
#define AMBIENT_CONTROLLERS_SSM_SCOPE

namespace ambient { 

    class scope {
    protected:
        typedef models::ssm::model model_type;
        typedef controllers::ssm::controller controller_type;
        scope(){}
    public:
        static int balance(int k, int max_k);
        static int permute(int k, const std::vector<int>& s);
       ~scope();
        scope(int r);
        scope(scope_t type);
        void set(int r);
        bool remote() const;
        bool local()  const;
        bool common() const;
        rank_t which()  const;
        scope_t type;
        bool dry;
        int factor;
        int round;
        int rank;
        ambient::locality state;
        controller_type* controller;
    };

    class base_scope : public scope {
    public:
        typedef typename scope::model_type model_type;
        base_scope();
        void schedule();
        void intend_read(models::ssm::revision* o);
        void intend_write(models::ssm::revision* o);
        mutable std::vector<int> stakeholders;
        mutable std::vector<int> scores;
    };

}

#endif
