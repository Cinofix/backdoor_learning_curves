from secml.array import CArray
from typing import Union, Tuple
import numpy as np


class Trigger:
    def __init__(
        self,
        input_size: Union[Tuple[int, int, int], Tuple[int, int]],
        trigger_type: str,
        n_triggers: int,
        position: Union[Tuple[int, int], str] = "btm-right",
        trigger_size: Tuple[int, int] = (3, 3),
        box: Tuple[float, float] = (0, 1),
        random_state: int = 999,
    ):
        if len(input_size) == 2:
            input_size = (1,) + input_size
        self.input_size = input_size
        self.trigger_type = trigger_type
        self.mask_type = position
        self.box = box
        self.n_distinct = n_triggers

        k, h, w = self.input_size
        self.trigger_size = (k,) + trigger_size
        self.random_state = random_state

        self.t = self._generate_trigger_pattern(trigger_type, box)
        self.mask = self._generate_mask(
            self.mask_type, self.input_size, self.trigger_size
        )

    def _generate_trigger_pattern(
        self, trigger_type: str, box: Tuple[float, float]
    ) -> CArray:
        k, t_h, t_w = self.trigger_size
        np.random.seed(self.random_state)

        # if isinstance(trigger_type, CArray):
        #    return trigger_type

        if not isinstance(trigger_type, str):
            raise ValueError(
                "Trigger trigger_content must be String [badnet, ones, or invisible]"
            )

        if not isinstance(self.n_distinct, int) or self.n_distinct <= 0:
            raise ValueError(
                "The number of distinct trigger must be a positive natural number."
            )
        elif trigger_type == "ones":
            trigger = np.ones((1, k, t_h, t_w))
            trigger = np.repeat(trigger, repeats=self.n_distinct, axis=0)
        elif trigger_type == "badnet" and self.n_distinct == 1:
            trigger = np.zeros((self.n_distinct, k, t_h, t_w))
            trigger[:] = np.random.rand(k, t_h, t_w)
        elif trigger_type == "badnet":
            trigger = np.random.rand(self.n_distinct, k, t_h, t_w)
        elif trigger_type == "invisible":
            k, h, w = self.input_size
            trigger = np.zeros((self.n_distinct, k, h, w))
            trigger[:, :, np.arange(0, h, 2), :] = 1
            trigger[:, :, :, np.arange(1, w, 2)] = 0
        else:
            raise ValueError(
                "Not supported trigger_content. Please use [full, distinct or random]"
            )
        trigger = self._adapt_to_box(trigger, box).reshape(self.n_distinct, -1)


        return CArray(trigger)

    @staticmethod
    def _adapt_to_box(trigger: np.array, box: Tuple[float, float]):
        lb, ub = box
        return lb + trigger * (ub - lb)

    def _generate_mask(self, mask_type, input_size, trigger_size):
        if isinstance(mask_type, CArray):
            return mask_type

        trigger_loc = mask_type

        if isinstance(trigger_loc, str) and trigger_loc != "full":
            trigger_loc = self._find_loc_coordinates(
                trigger_loc, input_size, trigger_size
            )
            trigger_mask = self._activate_mask_region(trigger_size, trigger_loc)

        elif isinstance(trigger_loc, str) and trigger_loc == "full":
            k, h, w = trigger_size
            trigger_mask = CArray.ones(k * h * w)
        else:
            trigger_loc = self._find_loc_coordinates(
                trigger_loc, input_size, trigger_size
            )
            trigger_mask = self._activate_mask_region(trigger_size, trigger_loc)

        return trigger_mask

    def _activate_mask_region(self, trigger_size, trigger_loc) -> CArray:
        # t_w, t_h = trigger_size
        # s_w, s_h = trigger_loc
        # trigger_mask[s_w : (s_w + t_w), s_h : (s_h + t_h)] = 1
        # swap row and columns
        k, h, w = self.input_size
        trigger_mask = np.zeros((k, h, w))

        k, t_h, t_w = trigger_size
        s_h, s_w = trigger_loc
        trigger_mask[:, s_h : (s_h + t_h), s_w : (s_w + t_w)] = 1
        trigger_mask = trigger_mask.flatten()
        return CArray(trigger_mask)

    @staticmethod
    def _find_loc_coordinates(position, input_size, trigger_size):
        k, i_h, i_w = input_size

        k, t_h, t_w = trigger_size
        if position == "btm-right":
            coordinates = i_h - t_h, i_w - t_w
        elif position == "btm-left":
            coordinates = i_w - t_w, 0
        elif position == "top-left":
            coordinates = 0, 0
        elif position == "top-right":
            coordinates = 0, i_w - t_w
        else:
            coordinates = i_h - t_h, i_w - t_w  # default bottom right
        return coordinates

    def trigger(self, x: CArray, y: int = None) -> CArray:
        """
        Trigger data sample x
        :param x: input sample
        :param y: base label when distinct triggers are used
        :return:
        """
        noise_trigger = self.get_trigger(y)
        if self.mask_type == "full":
            x_p = (x + noise_trigger).clip(0, 1)
        else:
            x_p = (1 - self.mask) * x + noise_trigger
        return x_p

    def get_trigger(self, y: int = None) -> np.array:
        trigger = self.t[y, :]

        mask = self.mask.tondarray()
        noise_trigger = np.zeros(self.input_size).flatten()
        noise_trigger[mask == 1] = trigger.tondarray().flatten()
        noise_trigger = CArray(noise_trigger)
        return noise_trigger

    def has_distinct_triggers(self):
        return True
