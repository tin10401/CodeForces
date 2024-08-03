#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ll unsigned long long
#define int long long
#define pb push_back
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define f first
#define s second
#define ar(x) array<int, x>
const static int INF = 1e18;
const static int mod = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int modExpo(int base, int exp)
{
    int res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}

void solve()
{
    set<int> s;
    int n;
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        int num;
        cin >> num;
        s.insert(num);
    }
    if(*s.rbegin() == 0)
    {
        cout << 0 << endl << endl;
        return;
    }
    int res = 0;
    vector<int> ans;
    while(res <= 40 && s.size() > 2 && *s.begin() != *s.rbegin())
    {
        auto it = prev(s.end());
        int diff = (*it + *(prev(it))) / 2;
        ans.push_back(diff);
        set<int> temp;
        for(auto& it : s) temp.insert(abs(it - diff));
        s = temp;
        res++;
    }
    if(s.size() == 1)
    {
       if(*s.rbegin() != 0)
       {
           res++;
           ans.push_back(*s.begin());
       }
    }
    else if(s.size() == 2)
    {
        int end = *s.rbegin(), start = *s.begin();
        if((end - start) % 2 != 0)
        {
            cout << -1 << endl;
            return;
        }
        if(start != 0)
        {
            res += 3;
            ans.push_back(start);
            ans.push_back((end - start) / 2), ans.push_back((end - start) / 2);
        }
        else
        {
            res += 2;
            ans.push_back(end / 2), ans.pb(end / 2);
        }
    }
    if(res > 40) 
    {
        cout << -1 << endl;
        return;
    }
    cout << res << endl;
    for(auto& it : ans) cout << it << " ";
    cout << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    int t;
    cin >> t;
    while(t--) solve();
    return 0;
}

