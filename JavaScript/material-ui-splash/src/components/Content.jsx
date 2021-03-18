
import React from "react";
import CoffeCard from "./CoffeCard";
import { Grid } from "@material-ui/core";
import coffeMakerList from "./static";

const Content = () => {
    const getCoffeMakerCard = coffeMakerObj => {
        return (
            <Grid item xs={12} sm={4}>
                <CoffeCard { ...coffeMakerObj} />
            </Grid>
        )
    }

    return (
        <Grid container space={2}>
            {coffeMakerList.map(coffeMakerObj => getCoffeMakerCard(coffeMakerObj))}
        </Grid>
    )
}

export default Content;