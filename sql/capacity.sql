select "Party_Name",
       count(*) as n_calls,
       min("Call_Start"),
       max("Call_Start")
from answered
group by "Party_Name"
order by n_calls desc;

