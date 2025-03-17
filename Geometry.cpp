struct Point {
    ld x, y;
    Point() : x(0), y(0) {}
    Point(ld x, ld y) : x(x), y(y) {}
    double distance(const Point &other) const {
        return std::sqrt((x - other.x) * (x - other.x) +
                         (y - other.y) * (y - other.y));
    }
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
        os << "(" << p.x << ", " << p.y << ")";
        return os;
    }

    ld slope_to(const Point &other) const {
        if (std::fabs(other.x - x) < eps) return std::numeric_limits<ld>::infinity();
        return (other.y - y) / (other.x - x);
    }
};

struct Circle {
    ld x, y, r;
    Circle(ld x = 0, ld y = 0, ld r = 0) : x(x), y(y), r(r) {}
};

struct Rectangle {
    ld x, y, w, h;
    Rectangle(ld x = 0, ld y = 0, ld w = 0, ld h = 0) : x(x), y(y), w(w), h(h) {}
};

bool circlesMergeWithinRect(const Circle &c1, const Circle &c2, const Rectangle &rect) { // check if the both circle intersect
                                                                                         // where the intersection is within the rectangle
    ld dx = c2.x - c1.x;
    ld dy = c2.y - c1.y;
    ld d = std::sqrt(dx * dx + dy * dy);
    if (d > c1.r + c2.r || d < fabsl(c1.r - c2.r)) return false;
    ld a = (c1.r * c1.r - c2.r * c2.r + d * d) / (2 * d);
    ld temp = c1.r * c1.r - a * a;
    if(temp < 0) temp = 0;
    ld h = std::sqrt(temp);
    ld px = c1.x + a * dx / d;
    ld py = c1.y + a * dy / d;
    if (fabs(h) < eps) {
        return (px >= rect.x && px <= rect.x + rect.w &&
                py >= rect.y && py <= rect.y + rect.h);
    } else {
        ld rx = -h * dy / d;
        ld ry = h * dx / d;
        ld ix1 = px + rx;
        ld iy1 = py + ry;
        ld ix2 = px - rx;
        ld iy2 = py - ry;
        bool inside1 = (ix1 >= rect.x && ix1 <= rect.x + rect.w &&
                        iy1 >= rect.y && iy1 <= rect.y + rect.h);
        bool inside2 = (ix2 >= rect.x && ix2 <= rect.x + rect.w &&
                        iy2 >= rect.y && iy2 <= rect.y + rect.h);
        return inside1 || inside2;
    }
}

bool circleRectangleIntersect(const Circle &c, const Rectangle &rect) { // check if circle intersect with the rectangle
    ld closestX = std::max(rect.x, std::min(c.x, rect.x + rect.w));
    ld closestY = std::max(rect.y, std::min(c.y, rect.y + rect.h));
    ld dx = c.x - closestX;
    ld dy = c.y - closestY;
    return (dx * dx + dy * dy) <= c.r * c.r;
}

bool circlesIntersect(const Circle &c1, const Circle &c2) { // check if two circle intersect
    ld dx = c1.x - c2.x;
    ld dy = c1.y - c2.y;
    ld distanceSq = dx * dx + dy * dy;
    ld radiusSum = c1.r + c2.r;
    return distanceSq <= radiusSum * radiusSum;
}

bool circleLineIntersect(const Circle &c, const Point &p1, const Point &p2) { // check if a circle intersect with a line
                                                                              // (x1, y1) to (x2, y2) inclusive
    long double cx = c.x, cy = c.y, cr = c.r;
    long double dx = p2.x - p1.x, dy = p2.y - p1.y;
    long double lenSq = dx * dx + dy * dy;
    if (lenSq == 0.0L) {
        long double distSq = (cx - p1.x) * (cx - p1.x) + (cy - p1.y) * (cy - p1.y);
        return distSq <= cr * cr;
    }
    long double t = ((cx - p1.x) * dx + (cy - p1.y) * dy) / lenSq;
    long double closestX, closestY;
    if (t < 0.0L) {
        closestX = p1.x;
        closestY = p1.y;
    } else if (t > 1.0L) {
        closestX = p2.x;
        closestY = p2.y;
    } else {
        closestX = p1.x + t * dx;
        closestY = p1.y + t * dy;
    }
    long double distSq = (cx - closestX) * (cx - closestX) + (cy - closestY) * (cy - closestY);
    return distSq <= cr * cr;
}

class CHT {
    public:
    int is_mx;
    vll m_slopes, b_intercepts;
    CHT(int is_mx) : is_mx(is_mx) {
        add_line(0, 0);
    }

    db cross(int i, int j, int k) {
        db A = (db)(1.00 * m_slopes[j] - m_slopes[i]) * (b_intercepts[k] - b_intercepts[i]);
        db B = (db)(1.00 * m_slopes[k] - m_slopes[i]) * (b_intercepts[j] - b_intercepts[i]);
        return is_mx ? A < B : A >= B;
    }

    void add(ll a, ll b) {
        if(is_mx) add_line(a, -b);
        else add_line(-a, b);
    }

    ll queries(ll x) {
        return is_mx ? -get(x) : get(x);
    }

    void add_line(ll slope, ll intercept) {
        m_slopes.push_back(slope);
        b_intercepts.push_back(intercept);
        while(m_slopes.size() >= 3 && cross(m_slopes.size() - 3, m_slopes.size() - 2, m_slopes.size() - 1)) {
            m_slopes.erase(m_slopes.end() - 2);
            b_intercepts.erase(b_intercepts.end() - 2);
        }
    }

    ll get(ll x) {
        if(m_slopes.empty()) return INF;
        int l = 0, r = m_slopes.size() - 1;
        while(l < r) {
            int mid = l + (r - l) / 2;
            ll f1 = m_slopes[mid] * x + b_intercepts[mid];
            ll f2 = m_slopes[mid + 1] * x + b_intercepts[mid + 1];
            if(f1 > f2) l = mid + 1;
            else r = mid;
        }
        return m_slopes[l] * x + b_intercepts[l];
    }
};

struct Line {
    mutable ll m, c, p;
    bool isQuery;
    bool operator<(const Line& o) const {
        if(o.isQuery)
            return p < o.p;
        return m < o.m;
    }
};

vvi rotate90(const vvi matrix) {
    int n = matrix.size(), m = matrix[0].size();
    vvi res(m, vi(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            res[j][n - 1 - i] = matrix[i][j];
    return res;
}

struct CHT : multiset<Line> {
    const ll inf = INF;
    int is_mx;
    ll div(ll a, ll b) {
        return a / b - ((a ^ b) < 0 && a % b); }
    bool isect(iterator x, iterator y) {
        if (y == end()) { x->p = inf; return false; }
        if (x->m == y->m) x->p = x->c > y->c ? inf : -inf;
        else x->p = div(y->c - x->c, x->m - y->m);
        return x->p >= y->p;
    }
    void add(ll m, ll c) {
        auto z = insert({m, c, 0, 0}), y = z++, x = y;
        while (isect(y, z)) z = erase(z);
        if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
        while ((y = x) != begin() && (--x)->p >= y->p)
            isect(x, erase(y));
    }
    ll query(ll x) {
        if(empty()) return inf;
        Line q; q.p = x, q.isQuery = 1;
        auto l = *lower_bound(q);
        return l.m * x + l.c;
    }
    // min will return -ans;
    // max will return ans;
    // max_normall is add(i, -dp)
    // min_normal is add(-i, dp)
};