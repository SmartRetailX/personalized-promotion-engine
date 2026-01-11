import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Tag,
  ShoppingBag,
  Clock,
  TrendingDown,
  Gift,
  Bell,
} from "lucide-react";
import { products, initialPromotions } from "../data/sampleData";

const CustomerDashboard = ({ customerId = 2 }) => {
  const navigate = useNavigate();
  // Simulate logged-in customer (Dinesh Silva - ID 2)
  const customer = {
    id: 2,
    name: "Dinesh Silva",
    email: "dinesh.s@email.com",
    region: "Colombo",
  };

  // Filter promotions for this customer
  const getAvailablePromotions = () => {
    return initialPromotions.filter((promo) => {
      if (promo.type === "bulk") return true;
      if (
        promo.type === "personalized" &&
        promo.targetCustomers.includes(customer.id)
      ) {
        // Check if still available
        return promo.usedCount < promo.availableCount;
      }
      return false;
    });
  };

  const [availablePromotions] = useState(getAvailablePromotions());
  const [redeemedPromotions, setRedeemedPromotions] = useState([]);

  const handleRedeemPromotion = (promo) => {
    setRedeemedPromotions([...redeemedPromotions, promo.id]);
    // In a real app, this would call the backend
    alert(
      `Promotion redeemed! Show this code at checkout: PROMO${promo.id}${customer.id}`
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-lg">SR</span>
            </div>
            <h1 className="text-xl font-bold text-gray-800">Smart RetailX</h1>
          </div>
          <div className="flex items-center gap-4">
            {/* Role Switcher */}
            <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => navigate("/admin")}
                className="px-3 py-1.5 text-gray-600 hover:bg-gray-200 rounded text-sm font-medium"
              >
                Admin
              </button>
              <button className="px-3 py-1.5 bg-green-500 text-white rounded text-sm font-medium">
                Customer
              </button>
            </div>
            <button className="p-2 hover:bg-gray-100 rounded-lg relative">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                <span className="text-white font-semibold text-sm">DS</span>
              </div>
              <div>
                <div className="text-sm font-semibold">{customer.name}</div>
                <div className="text-xs text-gray-500">CUSTOMER</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-2">
            My Promotions
          </h2>
          <p className="text-gray-500">
            Hello {customer.name}! Check out your exclusive offers
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white p-4 rounded-lg shadow-sm border">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                <Tag className="w-5 h-5 text-green-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-800">
                  {availablePromotions.length}
                </div>
                <div className="text-sm text-gray-600">Available Offers</div>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow-sm border">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                <ShoppingBag className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-800">
                  {redeemedPromotions.length}
                </div>
                <div className="text-sm text-gray-600">Redeemed</div>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow-sm border">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
                <TrendingDown className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-800">
                  {availablePromotions.reduce(
                    (sum, p) => sum + p.discountPercent,
                    0
                  ) / availablePromotions.length || 0}
                  %
                </div>
                <div className="text-sm text-gray-600">Avg. Discount</div>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow-sm border">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-orange-100 rounded-full flex items-center justify-center">
                <Gift className="w-5 h-5 text-orange-600" />
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-800">
                  LKR{" "}
                  {availablePromotions
                    .reduce(
                      (sum, p) => sum + (p.originalPrice - p.finalPrice),
                      0
                    )
                    .toFixed(0)}
                </div>
                <div className="text-sm text-gray-600">Total Savings</div>
              </div>
            </div>
          </div>
        </div>

        {/* Available Promotions */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
              <Tag className="w-6 h-6 text-blue-500" />
              Available Promotions for You
            </h3>
            <span className="text-sm text-gray-500">
              {availablePromotions.length} offer
              {availablePromotions.length !== 1 ? "s" : ""} available
            </span>
          </div>

          {availablePromotions.length === 0 ? (
            <div className="bg-white p-12 rounded-lg shadow-sm border text-center">
              <Gift className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h4 className="text-lg font-semibold text-gray-600 mb-2">
                No Promotions Available
              </h4>
              <p className="text-gray-500">
                Check back soon for exclusive offers!
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
              {availablePromotions.map((promo) => {
                const isRedeemed = redeemedPromotions.includes(promo.id);
                const isPersonalized = promo.type === "personalized";
                const remaining = isPersonalized
                  ? promo.availableCount - promo.usedCount
                  : null;
                const product = products.find((p) => p.id === promo.productId);

                return (
                  <div
                    key={promo.id}
                    className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden hover:shadow-lg transition-shadow"
                  >
                    {/* Product Image */}
                    <div className="relative">
                      <img
                        src={
                          product?.image ||
                          "https://images.unsplash.com/photo-1534723328310-e82dad3ee43f?w=400"
                        }
                        alt={promo.productName}
                        className="w-full h-36 object-cover"
                      />
                      {/* Discount Badge */}
                      <div className="absolute top-2 right-2 bg-red-500 text-white px-2 py-1 rounded-full shadow-lg">
                        <span className="text-xs font-bold">
                          {promo.discountPercent}% OFF
                        </span>
                      </div>
                    </div>

                    {/* Product Info */}
                    <div className="p-3">
                      <h4 className="text-base font-bold text-gray-800 mb-1">
                        {promo.productName}
                      </h4>
                      <p className="text-xs text-gray-600 mb-2">
                        {promo.brand} • {promo.category}
                      </p>

                      {/* Pricing */}
                      <div className="mb-3">
                        <div className="flex items-baseline gap-2">
                          <span className="text-xl font-bold text-green-600">
                            LKR {promo.finalPrice.toFixed(2)}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 line-through">
                          LKR {promo.originalPrice.toFixed(2)}
                        </div>
                        <div className="text-xs text-green-600 font-medium">
                          Save LKR{" "}
                          {(promo.originalPrice - promo.finalPrice).toFixed(2)}
                        </div>
                      </div>

                      {/* Limited Availability */}
                      {remaining && remaining <= 10 && (
                        <div className="bg-orange-50 border border-orange-200 rounded px-2 py-1.5 mb-2">
                          <div className="flex items-center gap-1 text-orange-700">
                            <Clock className="w-3 h-3" />
                            <span className="text-xs font-semibold">
                              Only {remaining} left!
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Redeem Button */}
                      <button
                        onClick={() => handleRedeemPromotion(promo)}
                        disabled={isRedeemed}
                        className={`w-full py-2 rounded-lg font-medium text-sm transition-all ${
                          isRedeemed
                            ? "bg-gray-200 text-gray-500 cursor-not-allowed"
                            : "bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white shadow-md hover:shadow-lg"
                        }`}
                      >
                        {isRedeemed ? "✓ Redeemed" : "Redeem Offer"}
                      </button>

                      {/* Valid Until */}
                      <div className="mt-2 pt-2 border-t border-gray-100 flex items-center justify-center text-xs text-gray-500">
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          Valid 7 days
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Redeemed Promotions */}
        {redeemedPromotions.length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <ShoppingBag className="w-5 h-5 text-green-500" />
              Redeemed Promotions
            </h3>
            <div className="space-y-3">
              {redeemedPromotions.map((promoId) => {
                const promo = availablePromotions.find((p) => p.id === promoId);
                if (!promo) return null;

                return (
                  <div
                    key={promoId}
                    className="flex items-center justify-between p-4 bg-green-50 border border-green-200 rounded-lg"
                  >
                    <div className="flex-1">
                      <div className="font-semibold text-gray-800">
                        {promo.productName}
                      </div>
                      <div className="text-sm text-gray-600">
                        {promo.discountPercent}% discount applied
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono text-sm bg-white px-3 py-1 rounded border border-gray-300">
                        PROMO{promo.id}
                        {customer.id}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Show at checkout
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Info Section */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h4 className="font-semibold text-blue-900 mb-2">💡 How it works</h4>
          <ul className="text-sm text-blue-800 space-y-2">
            <li>
              • Personalized offers are selected based on your shopping
              preferences
            </li>
            <li>
              • Redeem offers by clicking "Redeem Offer" and use the code at
              checkout
            </li>
            <li>
              • Limited quantity offers are available on a first-come,
              first-served basis
            </li>
            <li>• Keep shopping to unlock more exclusive deals!</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default CustomerDashboard;
