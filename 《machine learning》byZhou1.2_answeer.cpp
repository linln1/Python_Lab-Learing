#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stack>
using namespace std;

// 0表示*泛化情况
// 1 表示该属性下取第1种属性值，2，3同理
// 每三个一组，如000：***；123：三种属性分别取第1，2，3个属性值
// 一组中有几个0表示几属性泛化

static const char list[] = {
    0,0,0,                                                                            
    0,0,1,0,0,2,0,0,3,0,1,0,0,2,0,0,3,0,1,0,0,2,0,0, 
    0,1,1,0,1,2,0,1,3,0,2,1,0,2,2,0,2,3,0,3,1,0,3,2,0,3,3,
    1,0,1,1,0,2,1,0,3,2,0,1,2,0,2,2,0,3,
    1,1,0,1,2,0,1,3,0,2,1,0,2,2,0,2,3,0,
    1,1,1,1,1,2,1,1,3,1,2,1,1,2,2,1,2,3,1,3,1,1,3,2,1,3,3,
    2,1,1,2,1,2,2,1,3,2,2,1,2,2,2,2,2,3,2,3,1,2,3,2,2,3,3
};

class hypos {
public:
    virtual int insert(int cur) = 0;  //基类中的纯虚函数
};

//单个的假设类
// hypo_const 表示具体假设
class hypo: public hypos
{
public:
    hypo(int a, int b, int c)
    {
        hypo_const = 0;
        vector<char> p[3];
        // a = 0 表示第一种取泛化属性，需要把其包含的1，2两种具体属性都存入p容器中
        if(a == 0)   
        {
            p[0].push_back(1);
            p[0].push_back(2);
        }
        else p[0].push_back(a);

        if(b == 0)
        {
            p[1].push_back(1);
            p[1].push_back(2);
            p[1].push_back(3);
        }
        else p[1].push_back(b);

        if(c == 0)
        {
            p[2].push_back(1);
            p[2].push_back(2);
            p[2].push_back(3);
        }
        else p[2].push_back(c);

        for(unsigned int i = 0; i < p[0].size();i++)
            for(unsigned int j = 0; j < p[1].size(); j++)
                for(unsigned int k = 0; k < p[2].size(); k++)
                    // 最小 1 1 1：1*9+1*3+1 = 13
                    // 这里 -13为保证右移计数从0开始，每一种假设对应一位
                    // |= 表示按位或
                    // 对于每一种具体假设，基本只用一次 如 1 1 1 -> 0...01
                    // 对于有泛化的假设，则会用到如 1 1 0 
                    // 其p[2]位有1，2，3三种属性，需要将这三种假设对应位置为1
                    // 0...0 0001, 0...0 0010, 0...0 0100 按位或 = 0... 0 0111
                    hypo_const |= (1 << (p[0][i] * 9 + p[1][j] * 3 + p[2][k]) – 13);

    }

    int insert(int cur)
    {
        // 若 hypo_const & cur = 1，则可以入栈
        return (hypo_const & cur); 
    }

private:
    int hypo_const;  //表示具体假设
};

// 用于压入栈的派生类 用来实现非递归
// hypo_tmp 记录这个假设入栈时，带入了哪些具体假设，出栈时要还原 
// ptr 记录入栈时的位置

class hypo_ss: public hypos
{
public:
    hypo_ss(int _ptr, int tmp)
    {
        hypo_tmp = tmp;
        ptr = _ptr;
    }

    int insert(int cur)
    {   return 0;    }
    
    int hypo_tmp;
    int ptr;
};

// 用来循环遍历的类
// sum 各个长度的析合式各有多少种可能
// ss 用来实现非递归的栈 
// hypos_cur 当前没被包含的具体假设 初始值为0X3FFFF 
// hyposs 48个假设集合

class Traversal : public hypos
{
public:
    Traversal()
    {
        hypos_cur = 0x3ffff;
        for(int i = 0; i < 48; ++i)
        {
            // 见list[] : (0,0,0), (0,0,1) …
            // hypo(a, b, c) 表某个假设（属性分别为a,b,c）
            hyposs.push_back(hypo(list[3 * i], list[3 * i + 1], list[3 * i + 2])); 
        }
    }
    
    //循环顺序遍历的主体 
    //cur 初始的位置 设为0
    int insert(int cur)
    {
        int ptr = cur;    //当前指向的位置
        while(1)
        {
            //退出条件 当最后一个假设作为第一个入栈的元素 表示遍历完成
            if(ptr > 47 && !ss.size()) break;

            //回退条件 扫描到最后或者所有具体假设都被包含 
            if(hypos_cur == 0 || ptr > 47)
            {
                hypo_ss hypo_tmp = ss.top();
                hypos_cur ^= hypo_tmp.hypo_tmp;    //出栈异或
                ptr = hypo_tmp.ptr + 1;
                ss.pop();
                continue;
            }
            
            //入栈条件 如果该假设还有未被包含的具体假设 则入栈，
            // 并当前栈大小的计数加1
            if(int tmp = hyposs[ptr].insert(hypos_cur))
            {
                hypos_cur ^= tmp;
                ss.push(hypo_ss(ptr, tmp));
                if(sum.size() < ss.size())
                    sum.push_back(0);
                sum[ss.size() - 1]++;
            }
            ptr++;
        }
        return 1;
    }
    //输出各个长度的可能数
    void print()
    {
        for(unsigned int i = 0; i < sum.size(); ++i)
            printf("length %d : %d\n", i + 1, sum[i]);
    }

private:
    vector<int> sum;
    stack<hypo_ss> ss;
    int hypos_cur;
    vector<hypo> hyposs;
};

int main()
{
    Traversal traversal;
    traversal.insert(0);
    traversal.print();
    system("pause");
    return 0;
}

/* Output:
length 1 : 48
length 2 : 931
length 3 : 10332
length 4 : 72358
length 5 : 342057
length 6 : 1141603
length 7 : 2773332
length 8 : 4971915
length 9 : 6543060
length 10 : 6175660
length 11 : 4003914
length 12 : 1676233
length 13 : 422676
length 14 : 61884
length 15 : 5346
length 16 : 435
length 17 : 27
length 18 : 1
sh: 1: pause: not found
*/
