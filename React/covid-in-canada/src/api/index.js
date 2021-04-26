// import axios from 'axios'

const url = 'https://api.opencovid.ca/'

export const summary = async (region) => {
  const res = await fetch(`${url}summary?loc=${region}`)
  const data = await res.json().then(res => {
    return res
  })
  return data
  // axios.get(`${url}summary?loc=${region}`).then(res => {
  //   console.log('api response:', res)
  //   return res
  // })
}