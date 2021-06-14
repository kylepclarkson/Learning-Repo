import React from "react";

import Product from "./Product";

const ProductList = () => {

  return (
    <div className="col-md-8 col-sm-12">
      <h1>View Our Products Below</h1>
      <div className="row row-cols-md-3 row-cols-1 g-3">
        {products.map((product) => (
          <div className="col" key={product.id}>
            <Product product={product} />
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProductList;
