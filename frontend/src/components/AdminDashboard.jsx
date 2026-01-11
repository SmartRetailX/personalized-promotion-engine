import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ShoppingCart,
  TrendingUp,
  Package,
  AlertCircle,
  DollarSign,
  Users,
  Send,
  CheckCircle,
} from "lucide-react";
import {
  products,
  customers,
  getTopCustomersForProduct,
  initialPromotions,
} from "../data/sampleData.js";

const AdminDashboard = () => {
  const navigate = useNavigate();
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [discountPercent, setDiscountPercent] = useState(10);
  const [promotionType, setPromotionType] = useState("personalized");
  const [availableCount, setAvailableCount] = useState(50);
  const [targetCustomers, setTargetCustomers] = useState([]);
  const [promotions, setPromotions] = useState(initialPromotions);
  const [showSuccess, setShowSuccess] = useState(false);

  const totalRevenue = 1250000;
  const totalOrders = 4567;
  const activeAlerts = 3;
  const forecastAccuracy = 94.2;

  const handleProductSelect = (product) => {
    setSelectedProduct(product);
    if (promotionType === "personalized") {
      const topCustomers = getTopCustomersForProduct(product.id, 5);
      setTargetCustomers(topCustomers);
    }
  };

  const handlePromotionTypeChange = (type) => {
    setPromotionType(type);
    if (type === "bulk") {
      setTargetCustomers([]);
    } else if (selectedProduct) {
      const topCustomers = getTopCustomersForProduct(selectedProduct.id, 5);
      setTargetCustomers(topCustomers);
    }
  };

  const handleSendPromotion = () => {
    if (!selectedProduct) {
      alert("Please select a product first!");
      return;
    }

    const newPromotion = {
      id: promotions.length + 1,
      productId: selectedProduct.id,
      productName: selectedProduct.name,
      category: selectedProduct.category,
      brand: selectedProduct.brand,
      originalPrice: selectedProduct.price,
      discountPercent,
      finalPrice: selectedProduct.price * (1 - discountPercent / 100),
      type: promotionType,
      availableCount: promotionType === "personalized" ? availableCount : null,
      usedCount: 0,
      targetCustomers:
        promotionType === "personalized"
          ? targetCustomers.map((c) => c.id)
          : "all",
      createdAt: new Date().toISOString(),
      status: "active",
    };

    setPromotions([...promotions, newPromotion]);
    setShowSuccess(true);

    setTimeout(() => {
      setShowSuccess(false);
      setSelectedProduct(null);
      setDiscountPercent(10);
      setTargetCustomers([]);
    }, 3000);
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
            <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
              <button className="px-3 py-1.5 bg-blue-500 text-white rounded text-sm font-medium">
                Admin
              </button>
              <button
                onClick={() => navigate("/customer")}
                className="px-3 py-1.5 text-gray-600 hover:bg-gray-200 rounded text-sm font-medium transition"
              >
                Customer
              </button>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <span className="text-white font-semibold text-sm">SA</span>
              </div>
              <div>
                <div className="text-sm font-semibold">System Admin</div>
                <div className="text-xs text-gray-500">ADMIN</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Dashboard Overview */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-800 mb-2">
            Dashboard Overview
          </h2>
          <p className="text-gray-500">Welcome back, System Admin!</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-sm border col-span-2">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">Total Revenue</span>
              <DollarSign className="w-5 h-5 text-green-500" />
            </div>
            <div className="text-2xl font-bold text-gray-800">
              LKR {totalRevenue.toLocaleString()}.00
            </div>
            <div className="text-sm text-green-600 mt-1">
              ↑ 12.5% vs last period
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">Total Orders</span>
              <ShoppingCart className="w-5 h-5 text-blue-500" />
            </div>
            <div className="text-2xl font-bold text-gray-800">
              {totalOrders.toLocaleString()}
            </div>
            <div className="text-sm text-green-600 mt-1">
              ↑ 8.3% vs last period
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">Active Alerts</span>
              <AlertCircle className="w-5 h-5 text-red-500" />
            </div>
            <div className="text-2xl font-bold text-gray-800">
              {activeAlerts}
            </div>
            <div className="text-sm text-red-600 mt-1">
              ↓ 15.2% vs last period
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">Forecast Accuracy</span>
              <TrendingUp className="w-5 h-5 text-purple-500" />
            </div>
            <div className="text-2xl font-bold text-gray-800">
              {forecastAccuracy}%
            </div>
            <div className="text-sm text-green-600 mt-1">
              ↑ 2.1% vs last period
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">Total Products</span>
              <Package className="w-5 h-5 text-indigo-500" />
            </div>
            <div className="text-2xl font-bold text-gray-800">20</div>
            <div className="text-sm text-gray-500 mt-1">
              0.0% vs last period
            </div>
          </div>
        </div>

        {/* Promotion Creation Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Create Promotion Form */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <Send className="w-5 h-5 text-blue-500" />
              Create Promotion
            </h3>

            {/* Product Selection */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Product
              </label>
              <select
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={selectedProduct?.id || ""}
                onChange={(e) => {
                  const product = products.find(
                    (p) => p.id === parseInt(e.target.value)
                  );
                  handleProductSelect(product);
                }}
              >
                <option value="">Choose a product...</option>
                {products.map((product) => (
                  <option key={product.id} value={product.id}>
                    {product.name} - {product.brand} (LKR{" "}
                    {product.price.toFixed(2)})
                  </option>
                ))}
              </select>
            </div>

            {selectedProduct && (
              <div className="bg-blue-50 p-4 rounded-lg mb-4">
                <div className="text-sm">
                  <div className="font-semibold text-gray-800">
                    {selectedProduct.name}
                  </div>
                  <div className="text-gray-600">
                    Category: {selectedProduct.category}
                  </div>
                  <div className="text-gray-600">
                    Brand: {selectedProduct.brand}
                  </div>
                  <div className="text-lg font-bold text-blue-600 mt-2">
                    LKR {selectedProduct.price.toFixed(2)}
                  </div>
                </div>
              </div>
            )}

            {/* Discount Percentage */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Discount Percentage: {discountPercent}%
              </label>
              <input
                type="range"
                min="5"
                max="50"
                step="5"
                value={discountPercent}
                onChange={(e) => setDiscountPercent(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>5%</span>
                <span>25%</span>
                <span>50%</span>
              </div>
            </div>

            {selectedProduct && (
              <div className="bg-green-50 p-3 rounded-lg mb-4">
                <div className="text-sm text-gray-600">Final Price:</div>
                <div className="text-xl font-bold text-green-600">
                  LKR{" "}
                  {(
                    selectedProduct.price *
                    (1 - discountPercent / 100)
                  ).toFixed(2)}
                </div>
                <div className="text-xs text-gray-500">
                  Save LKR{" "}
                  {((selectedProduct.price * discountPercent) / 100).toFixed(2)}
                </div>
              </div>
            )}

            {/* Promotion Type Selection */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Promotion Type
              </label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => handlePromotionTypeChange("personalized")}
                  className={`px-4 py-3 rounded-lg border-2 font-medium transition-all flex flex-col items-center gap-2 ${
                    promotionType === "personalized"
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-300 bg-white text-gray-700 hover:border-blue-300"
                  }`}
                >
                  <Users className="w-5 h-5" />
                  <span>Personalized</span>
                </button>
                <button
                  type="button"
                  onClick={() => handlePromotionTypeChange("bulk")}
                  className={`px-4 py-3 rounded-lg border-2 font-medium transition-all flex flex-col items-center gap-2 ${
                    promotionType === "bulk"
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-300 bg-white text-gray-700 hover:border-blue-300"
                  }`}
                >
                  <Package className="w-5 h-5" />
                  <span>Bulk</span>
                </button>
              </div>
            </div>

            {/* Available Count - Only for Personalized */}
            {promotionType === "personalized" && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Available Discounts
                </label>
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={availableCount}
                  onChange={(e) => setAvailableCount(parseInt(e.target.value))}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}

            {/* Send Promotion Button */}
            <button
              onClick={handleSendPromotion}
              disabled={!selectedProduct}
              className={`w-full py-3 rounded-lg font-semibold flex items-center justify-center gap-2 transition-all ${
                selectedProduct
                  ? "bg-blue-500 hover:bg-blue-600 text-white"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }`}
            >
              <Send className="w-5 h-5" />
              Send Promotion
            </button>

            {showSuccess && (
              <div className="mt-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded-lg flex items-center gap-2">
                <CheckCircle className="w-5 h-5" />
                <span>Promotion sent successfully!</span>
              </div>
            )}
          </div>

          {/* Target Customers Preview */}
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
              <Users className="w-5 h-5 text-purple-500" />
              {promotionType === "personalized"
                ? "Target Customers (AI Predicted)"
                : "Target: All Customers"}
            </h3>

            {promotionType === "bulk" && (
              <div className="text-center py-12">
                <Package className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">
                  This promotion will be sent to all {customers.length}{" "}
                  customers
                </p>
                <p className="text-sm text-gray-400 mt-2">
                  No AI targeting - bulk distribution to everyone
                </p>
              </div>
            )}

            {promotionType === "personalized" &&
              targetCustomers.length === 0 && (
                <div className="text-center py-12">
                  <Users className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">
                    Select a product to see target customers
                  </p>
                  <p className="text-sm text-gray-400 mt-2">
                    AI will predict customers with highest buying probability
                  </p>
                </div>
              )}

            {targetCustomers.length > 0 && (
              <div className="space-y-3">
                <div className="bg-purple-50 p-3 rounded-lg mb-4">
                  <div className="text-sm font-medium text-purple-900">
                    Top {targetCustomers.length} Customers Selected
                  </div>
                  <div className="text-xs text-purple-700">
                    Based on purchase history and ML predictions
                  </div>
                </div>

                {targetCustomers.map((customer, index) => (
                  <div
                    key={customer.id}
                    className="bg-gray-50 p-4 rounded-lg border border-gray-200"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="font-semibold text-gray-800 flex items-center gap-2">
                          <span className="bg-blue-500 text-white w-6 h-6 rounded-full flex items-center justify-center text-xs">
                            {index + 1}
                          </span>
                          {customer.name}
                        </div>
                        <div className="text-sm text-gray-600 mt-1">
                          {customer.email}
                        </div>
                        <div className="flex gap-3 mt-2 text-xs">
                          <span className="bg-gray-200 px-2 py-1 rounded">
                            {customer.region}
                          </span>
                          <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded">
                            Age: {customer.age}
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-gray-500">
                          Buy Probability
                        </div>
                        <div className="text-lg font-bold text-green-600">
                          {customer.buyingProbability}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Active Promotions */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-xl font-bold text-gray-800 mb-6">
            Active Promotions
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase">
                    Product
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase">
                    Type
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase">
                    Discount
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase">
                    Price
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase">
                    Usage
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {promotions.map((promo) => (
                  <tr key={promo.id} className="hover:bg-gray-50">
                    <td className="px-4 py-4">
                      <div className="font-medium text-gray-800">
                        {promo.productName}
                      </div>
                      <div className="text-sm text-gray-500">
                        {promo.brand} • {promo.category}
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-medium ${
                          promo.type === "personalized"
                            ? "bg-purple-100 text-purple-700"
                            : "bg-blue-100 text-blue-700"
                        }`}
                      >
                        {promo.type}
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <span className="font-bold text-green-600">
                        {promo.discountPercent}% OFF
                      </span>
                    </td>
                    <td className="px-4 py-4">
                      <div className="text-sm text-gray-500 line-through">
                        LKR {promo.originalPrice.toFixed(2)}
                      </div>
                      <div className="font-bold text-gray-800">
                        LKR {promo.finalPrice.toFixed(2)}
                      </div>
                    </td>
                    <td className="px-4 py-4">
                      {promo.type === "personalized" ? (
                        <div className="text-sm">
                          <div className="font-medium">
                            {promo.usedCount} / {promo.availableCount}
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                            <div
                              className="bg-blue-500 h-2 rounded-full transition-all"
                              style={{
                                width: `${
                                  (promo.usedCount / promo.availableCount) * 100
                                }%`,
                              }}
                            />
                          </div>
                        </div>
                      ) : (
                        <div className="text-sm font-medium">
                          {promo.usedCount} redeemed
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-4">
                      <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                        {promo.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
