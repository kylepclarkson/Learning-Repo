

const url = 'https://api.opencovid.ca/'

export const summary = async (region) => {

    const res = await fetch(`${url}?loc=${region}`)
    return await res.json()
}